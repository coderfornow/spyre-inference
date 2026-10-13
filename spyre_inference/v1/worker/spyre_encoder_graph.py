# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Upstream's encoder budget manager, compiling each budget at warmup instead of
capturing a CUDA graph. Budgets, packing, padding and the per-item split are upstream's."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.logger import init_logger
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import STICK

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph

    from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin

logger = init_logger(__name__)


class SpyreEncoderGraphManager(EncoderCudaGraphManager):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        dtype: torch.dtype,
        model: SpyreVisionEncoderMixin,
    ) -> None:
        super().__init__(vllm_config, device, dtype, cast("SupportsEncoderCudaGraph", model))
        if self.use_dp:
            raise NotImplementedError(
                "mm_encoder_tp_mode='data' is not supported on Spyre; the vision "
                "encoder runs tensor-parallel with the decoder."
            )
        # Every L must be whole sticks of rows.
        step = math.lcm(STICK, model.rows_per_token) // model.rows_per_token

        def aligned(budgets: list[int]) -> list[int]:
            return sorted({-(-b // step) * step if b > 0 else b for b in budgets})

        self.token_budgets = aligned(self.token_budgets)
        self.path_token_budgets = {p: aligned(b) for p, b in self.path_token_budgets.items()}
        if self.token_budgets[-1] < model.max_tokens_per_item:
            raise ValueError(
                f"largest encoder budget {self.token_budgets[-1]} is below the largest "
                f"image ({model.max_tokens_per_item} tokens), which would then compile "
                "mid-request. Raise compilation_config.encoder_cudagraph_token_budgets."
            )
        self._staging: dict[str, dict[int, dict[str, torch.Tensor]]] = {}

    def _capture_budget_graph(self, token_budget: int, path: str = "default") -> None:
        values = self.model.prepare_encoder_cudagraph_capture_inputs(
            token_budget,
            self.max_batch_size,
            self.max_frames_per_batch,
            torch.device("cpu"),
            self.dtype,
            path,
        ).values
        t0 = time.time()
        with torch.inference_mode():
            self.model.encoder_cudagraph_forward(self._upload(values), path=path)
        logger.info("Vision encoder budget %d compiled in %.1fs.", token_budget, time.time() - t0)
        self._staging.setdefault(path, {})[token_budget] = values

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        path: str = "default",
    ) -> torch.Tensor | None:
        staging = self._staging.get(path, {}).get(token_budget)
        num_items = len(self._get_item_specs(mm_kwargs))
        if staging is None:
            self.graph_misses += num_items
            return None
        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs, self.max_batch_size, self.max_frames_per_batch, path
        )
        for key, buf in staging.items():
            src = replay.values.get(key)
            if src is not None:
                self.config.padding_logics.get(key, self._copy_padded_buffer)(buf, src)
        with torch.inference_mode():
            output = self.model.encoder_cudagraph_forward(self._upload(staging), path=path)
        self.graph_hits += num_items
        return output

    def _upload(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: convert(value, device=self.device) for key, value in values.items()}
