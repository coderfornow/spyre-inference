# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the rope permutation builder and the `rope_perm_matrix` escape valve.

The builder backs both towers' warmed caches; the op is the fallback for a
permutation no warm call can reach, and only the on-card case proves it lowers.
"""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.multimodal.utils import build_rope_perm, rope_perm_op

DIM = 64


def test_pair_permutation_swaps_adjacent_elements():
    x = torch.randn(3, DIM, dtype=torch.float16)
    expected = x.reshape(3, DIM // 2, 2).flip(-1).reshape(3, DIM)
    torch.testing.assert_close(x @ build_rope_perm("pair", DIM, torch.float16), expected)


def test_half_swap_permutation_matches_the_slice_it_replaces():
    x = torch.randn(3, DIM, dtype=torch.float16)
    half = DIM // 2
    expected = torch.cat([x[..., half:], x[..., :half]], dim=-1)
    torch.testing.assert_close(x @ build_rope_perm("half_swap", DIM, torch.float16), expected)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown rope permutation kind"):
        build_rope_perm("quarter_turn", DIM, torch.float16)


def test_op_reuses_its_cached_matrix():
    """One host build per key: the op runs on every forward it appears in."""
    first = rope_perm_op("pair", DIM, torch.float16, torch.device("cpu"))
    assert rope_perm_op("pair", DIM, torch.float16, torch.device("cpu")) is first


def test_op_lowers_inside_a_compiled_graph_on_spyre():
    """What the op buys: a permutation assembled where no warm call could reach it.

    Inline, the same build leaves a CPU buffer in the graph and layout propagation
    rejects it (`... does not have FixedTiledLayout`).
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    device = torch.device("spyre")

    def rotate(x):
        return x @ rope_perm_op("pair", DIM, torch.float16, x.device)

    torch.manual_seed(17)
    x = torch.randn(8, DIM, dtype=torch.float16)
    expected = rotate(x)

    torch._dynamo.reset()
    compiled = torch.compile(rotate, backend="inductor", fullgraph=True, dynamic=False)
    actual = compiled(x.to(device))

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)
