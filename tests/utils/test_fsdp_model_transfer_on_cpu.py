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

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from torch.distributed.fsdp._flat_param import FlatParameter

from verl.utils import fsdp_utils


@pytest.mark.parametrize("device_name, non_blocking", [("npu", False), ("cuda", True)])
@pytest.mark.parametrize("empty_cache", [False, True])
def test_offload_fsdp_model_copy_policy_and_shard_aliases(monkeypatch, device_name, non_blocking, empty_cache):
    model = Mock(spec=fsdp_utils.FSDP)
    model._is_root = True
    native_offload_handle = Mock(_offload_params=True)
    handles = []
    original_shards = []
    for size in (4, 8):
        flat_param = FlatParameter(torch.arange(size, dtype=torch.float32))
        flat_param._local_shard = flat_param.data
        original_shards.append(flat_param._local_shard)

        def move_to_cpu(*args, param=flat_param, **kwargs):
            param.data = param.data.clone()

        handles.append(
            SimpleNamespace(_offload_params=False, flat_param=flat_param, flat_param_to=Mock(side_effect=move_to_cpu))
        )
    model._all_handles = [handles[0], native_offload_handle, handles[1]]
    lazy_init = Mock()
    device_module = Mock()
    calls = Mock()
    for index, handle in enumerate(handles):
        calls.attach_mock(handle.flat_param_to, f"copy_{index}")
    calls.attach_mock(device_module.empty_cache, "empty_cache")
    monkeypatch.setattr(fsdp_utils, "_lazy_init", lazy_init)
    monkeypatch.setattr(fsdp_utils, "get_device_name", lambda: device_name)
    monkeypatch.setattr(fsdp_utils, "get_torch_device", lambda: device_module)

    fsdp_utils.offload_fsdp_model_to_cpu(model, empty_cache=empty_cache)

    lazy_init.assert_called_once_with(model, model)
    native_offload_handle.flat_param_to.assert_not_called()
    expected = [
        call.copy_0(torch.device("cpu"), non_blocking=non_blocking),
        call.copy_1(torch.device("cpu"), non_blocking=non_blocking),
    ]
    if empty_cache:
        expected.append(call.empty_cache())
    assert calls.mock_calls == expected
    for handle, original in zip(handles, original_shards, strict=True):
        param = handle.flat_param
        torch.testing.assert_close(param.data, original)
        assert param.data.data_ptr() != original.data_ptr()
        assert param.data.data_ptr() == param._local_shard.data_ptr()
        assert param.data is not param._local_shard


@pytest.mark.parametrize("device_name, non_blocking", [("npu", False), ("cuda", True)])
def test_offload_fsdp_optimizer_copy_policy(monkeypatch, device_name, non_blocking):
    params = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    states = []
    for _ in params:
        tensors = {key: Mock(spec=torch.Tensor) for key in ("step", "exp_avg", "exp_avg_sq")}
        states.append({**tensors, "metadata": 7})
    optimizer = SimpleNamespace(
        param_groups=[{"params": [param]} for param in params], state=dict(zip(params, states, strict=True))
    )
    monkeypatch.setattr(fsdp_utils, "get_device_name", lambda: device_name)
    original_states = [state.copy() for state in states]

    fsdp_utils.offload_fsdp_optimizer(optimizer)

    for state, original in zip(states, original_states, strict=True):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            original[key].to.assert_called_once_with("cpu", non_blocking=non_blocking)
            assert state[key] is original[key].to.return_value
        assert state["metadata"] == 7


def test_offload_fsdp_optimizer_without_state():
    optimizer = Mock(state={})

    fsdp_utils.offload_fsdp_optimizer(optimizer)

    assert optimizer.mock_calls == []
