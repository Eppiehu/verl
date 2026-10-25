# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from unittest.mock import Mock

import pytest
import torch

from verl.utils import fsdp_utils


@pytest.mark.parametrize("device_name,non_blocking", [("cuda", True), ("npu", False)])
@pytest.mark.parametrize("empty_cache", [False, True])
def test_offload_fsdp1_model_preserves_shards(monkeypatch, device_name, non_blocking, empty_cache):
    model = Mock(spec=fsdp_utils.FSDP)
    model._is_root = True
    flat_param = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    flat_param._local_shard = flat_param.data
    original_shard = flat_param._local_shard
    handle = Mock(_offload_params=False, flat_param=flat_param)
    managed_handle = Mock(_offload_params=True)
    model._all_handles = [managed_handle, handle]
    device_module = Mock()
    lazy_init = Mock()
    monkeypatch.setattr(fsdp_utils, "_lazy_init", lazy_init)
    monkeypatch.setattr(fsdp_utils, "get_device_name", lambda: device_name)
    monkeypatch.setattr(fsdp_utils, "get_torch_device", lambda: device_module)

    def flat_param_to(device, *, non_blocking):
        # Allocate a new CPU shard to exercise the alias update without an accelerator.
        flat_param.data = flat_param.to(device, non_blocking=non_blocking, copy=True)

    handle.flat_param_to.side_effect = flat_param_to

    fsdp_utils.offload_fsdp_model_to_cpu(model, empty_cache=empty_cache)

    lazy_init.assert_called_once_with(model, model)
    handle.flat_param_to.assert_called_once_with(torch.device("cpu"), non_blocking=non_blocking)
    managed_handle.flat_param_to.assert_not_called()
    torch.testing.assert_close(flat_param, original_shard)
    assert flat_param.data_ptr() != original_shard.data_ptr()
    assert flat_param._local_shard.data_ptr() == flat_param.data_ptr()
    assert flat_param._local_shard is not flat_param.data
    assert device_module.empty_cache.call_count == int(empty_cache)


@pytest.mark.parametrize("device_name,non_blocking", [("cuda", True), ("npu", False)])
def test_offload_optimizer_preserves_state(monkeypatch, device_name, non_blocking):
    parameter = object()
    exp_avg = Mock(spec=torch.Tensor)
    exp_avg_sq = Mock(spec=torch.Tensor)
    exp_avg.to.return_value = torch.tensor([1.0, 2.0])
    exp_avg_sq.to.return_value = torch.tensor([3.0, 4.0])
    state = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "step": 2}
    optimizer = Mock(state={parameter: state}, param_groups=[{"params": [parameter]}])
    monkeypatch.setattr(fsdp_utils, "get_device_name", lambda: device_name)

    fsdp_utils.offload_fsdp_optimizer(optimizer)

    exp_avg.to.assert_called_once_with("cpu", non_blocking=non_blocking)
    exp_avg_sq.to.assert_called_once_with("cpu", non_blocking=non_blocking)
    assert state["exp_avg"] is exp_avg.to.return_value
    assert state["exp_avg_sq"] is exp_avg_sq.to.return_value
    assert state["step"] == 2


def test_offload_optimizer_without_state():
    optimizer = Mock(state={})

    fsdp_utils.offload_fsdp_optimizer(optimizer)

    assert optimizer.mock_calls == []
