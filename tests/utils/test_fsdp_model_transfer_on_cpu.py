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

from verl.utils import fsdp_utils


@pytest.mark.parametrize("device_name, non_blocking", [("cuda", True), ("npu", False)])
@pytest.mark.parametrize("empty_cache", [False, True])
def test_offload_fsdp_model_to_cpu_copy_policy(monkeypatch, device_name, non_blocking, empty_cache):
    model = Mock(spec=fsdp_utils.FSDP)
    model._is_root = True
    flat_param = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    flat_param._local_shard = flat_param.detach()
    original_shard = flat_param._local_shard

    def copy_to_cpu(device, *, non_blocking):
        # A separate allocation also lets us check that _local_shard is updated.
        flat_param.data = flat_param.to(device, non_blocking=non_blocking, copy=True)

    handle = SimpleNamespace(_offload_params=False, flat_param=flat_param, flat_param_to=Mock(side_effect=copy_to_cpu))
    offloaded_handle = Mock(_offload_params=True)
    model._all_handles = [offloaded_handle, handle]
    device_module = Mock()
    calls = Mock()
    calls.attach_mock(handle.flat_param_to, "flat_param_to")
    calls.attach_mock(device_module.empty_cache, "empty_cache")
    lazy_init = Mock()
    monkeypatch.setattr(fsdp_utils, "_lazy_init", lazy_init)
    monkeypatch.setattr(fsdp_utils, "get_device_name", lambda: device_name)
    monkeypatch.setattr(fsdp_utils, "get_torch_device", lambda: device_module)

    fsdp_utils.offload_fsdp_model_to_cpu(model, empty_cache=empty_cache)

    lazy_init.assert_called_once_with(model, model)
    offloaded_handle.flat_param_to.assert_not_called()
    expected = [call.flat_param_to(torch.device("cpu"), non_blocking=non_blocking)]
    if empty_cache:
        expected.append(call.empty_cache())
    assert calls.mock_calls == expected
    torch.testing.assert_close(flat_param, original_shard)
    assert flat_param.device.type == "cpu"
    assert flat_param._local_shard.data_ptr() == flat_param.data.data_ptr()
    assert flat_param._local_shard.data_ptr() != original_shard.data_ptr()
    assert flat_param._local_shard is not flat_param.data
