"""CPU tests for EP-aware clip_grad_norm (mixed Tensor / DTensor, no GPU)."""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from axolotl.monkeypatch.accelerate.parallelism_config import (
    _all_reduce_scalar,
    _dtensor_replicate_scale,
    _ep_aware_clip_grad_norm,
    _grads_need_mixed_clip,
    _should_use_ep_aware_clip,
)


class _GradHolder:
    """Stand-in for a parameter; clip only reads ``.grad``."""

    def __init__(self, grad):
        self.grad = grad


def _params_with_grads(*tensors: torch.Tensor) -> list[torch.nn.Parameter]:
    params = []
    for t in tensors:
        p = torch.nn.Parameter(torch.zeros_like(t))
        p.grad = t.clone()
        params.append(p)
    return params


class TestEpAwareClipPlainTensors:
    def test_matches_stock_clip(self):
        grads = [torch.arange(6, dtype=torch.float32), torch.ones(4)]
        ours = _params_with_grads(*grads)
        stock = _params_with_grads(*grads)
        total = _ep_aware_clip_grad_norm(ours, max_norm=1.0)
        expected = torch.nn.utils.clip_grad_norm_(stock, max_norm=1.0)
        assert torch.allclose(total, expected, rtol=1e-5, atol=1e-5)
        for a, b in zip(ours, stock, strict=True):
            assert torch.allclose(a.grad, b.grad, rtol=1e-5, atol=1e-5)

    def test_inf_norm(self):
        params = _params_with_grads(torch.tensor([3.0, -4.0]), torch.tensor([1.0]))
        total = _ep_aware_clip_grad_norm(params, max_norm=2.0, norm_type=float("inf"))
        assert total.item() == pytest.approx(4.0)
        # 2/4 = 0.5
        assert torch.allclose(params[0].grad, torch.tensor([1.5, -2.0]))

    def test_empty_grads(self):
        p = torch.nn.Parameter(torch.ones(2))
        assert _ep_aware_clip_grad_norm([p], 1.0).item() == 0.0

    def test_cpu_scalar_reduce_noop_without_dist(self):
        acc = torch.tensor(3.0)
        out = _all_reduce_scalar(acc, None)
        assert out.item() == 3.0


class TestClipDispatch:
    def test_pure_ep_uses_aware_clip(self):
        accel = SimpleNamespace(
            parallelism_config=SimpleNamespace(ep_enabled=True, dp_shard_enabled=False)
        )
        p = torch.nn.Parameter(torch.ones(2))
        p.grad = torch.ones(2)
        assert _should_use_ep_aware_clip(accel, [p]) is True

    def test_no_ep_plain_tensors_use_stock(self):
        accel = SimpleNamespace(parallelism_config=SimpleNamespace(ep_enabled=False))
        p = torch.nn.Parameter(torch.ones(2))
        p.grad = torch.ones(2)
        assert _should_use_ep_aware_clip(accel, [p]) is False
        assert _grads_need_mixed_clip([p]) is False

    def test_missing_parallelism_config_plain(self):
        accel = SimpleNamespace(parallelism_config=None)
        p = torch.nn.Parameter(torch.ones(2))
        p.grad = torch.ones(2)
        assert _should_use_ep_aware_clip(accel, [p]) is False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def gloo_world1():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.usefixtures("gloo_world1")
class TestMixedTensorDTensorClip:
    def test_stock_clip_raises_mixed_types(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        plain = _GradHolder(torch.ones(4))
        wrapped = _GradHolder(DTensor.from_local(torch.ones(4), mesh, [Replicate()]))
        with pytest.raises(RuntimeError):
            torch.nn.utils.clip_grad_norm_([plain, wrapped], 1.0)

    def test_ep_aware_clip_accepts_mixed_types(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        plain = _GradHolder(torch.ones(4))
        wrapped = _GradHolder(DTensor.from_local(torch.ones(4), mesh, [Replicate()]))
        assert _grads_need_mixed_clip([plain, wrapped]) is True
        total = _ep_aware_clip_grad_norm([plain, wrapped], max_norm=1.0)
        assert torch.isfinite(total)
        # ||concat(ones(4), ones(4))||_2 = sqrt(8); coef = 1/sqrt(8)
        expected = 1.0 / (8.0**0.5)
        assert torch.allclose(plain.grad, torch.full((4,), expected), rtol=1e-5)
        assert torch.allclose(
            wrapped.grad.to_local(), torch.full((4,), expected), rtol=1e-5
        )

    def test_mixed_meshes_need_aware_clip(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate

        mesh_a = init_device_mesh("cpu", (1,), mesh_dim_names=("a",))
        mesh_b = init_device_mesh("cpu", (1,), mesh_dim_names=("b",))
        pa = _GradHolder(DTensor.from_local(torch.ones(2), mesh_a, [Replicate()]))
        pb = _GradHolder(DTensor.from_local(torch.ones(2), mesh_b, [Replicate()]))
        assert _grads_need_mixed_clip([pa, pb]) is True
        total = _ep_aware_clip_grad_norm([pa, pb], max_norm=10.0)
        assert torch.isfinite(total)

    def test_replicate_scale_is_one_on_size1_mesh(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        g = DTensor.from_local(torch.ones(3), mesh, [Replicate()])
        assert _dtensor_replicate_scale(g) == pytest.approx(1.0)
        assert _dtensor_replicate_scale(torch.ones(3)) == pytest.approx(1.0)
