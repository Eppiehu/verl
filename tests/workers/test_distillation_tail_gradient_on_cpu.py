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
"""Numerical checks for the Megatron coarse-grained distillation backward."""

import pytest
import torch

from verl.trainer.distillation.tail_kl import compute_tail_aware_logit_gradient


@pytest.mark.parametrize("shard_size", [1, 2, 4])
@pytest.mark.parametrize("tail_mass", [2.0**-16, 2.0**-22], ids=["active", "clamped"])
def test_concentrated_student_matches_coarse_kl_autograd(shard_size, tail_mass):
    eps = 1e-6
    # Exact binary fractions isolate backward cancellation from mass-rounding error.
    student = torch.tensor([0.5, tail_mass / 2, 0.5 - tail_mass, tail_mass / 2])
    ids = torch.tensor([0, 2])
    teacher = torch.tensor([0.4999, 0.3])
    logits = student.double().log().requires_grad_()
    probs = logits.softmax(dim=-1)
    teacher64 = teacher.double()
    teacher_tail = 1 - teacher64.sum()
    loss = (teacher64 * (teacher64.log() - probs[ids].log())).sum()
    loss += teacher_tail * (teacher_tail.log() - probs[[1, 3]].sum().clamp_min(eps).log())
    (expected,) = torch.autograd.grad(loss, logits)

    shards = []
    for start in range(0, student.numel(), shard_size):
        local = (ids >= start) & (ids < start + shard_size)
        shards.append(
            compute_tail_aware_logit_gradient(
                student_probs=student[start : start + shard_size],
                teacher_topk_probs=teacher.masked_fill(~local, 0),
                teacher_topk_indices=(ids - start).masked_fill(~local, 0),
                teacher_topk_mask=local,
                student_topk_mass=student[ids].sum(),
                teacher_topk_mass=teacher.sum(),
                tail_mass_eps=eps,
            )
        )
    torch.testing.assert_close(torch.cat(shards), expected.float(), atol=1e-7, rtol=1e-5)
