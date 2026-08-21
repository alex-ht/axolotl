"""CPU-scalar all-reduce routing for CPU-offloaded gradient clipping."""

import pytest
import torch
import torch.distributed as dist

from axolotl.utils.gradient_clipping import (
    _all_reduce_scalar,
    _backend_rejects_cpu_tensors,
)


def _stub_dist(monkeypatch, backend: str, *, cuda_available: bool = True):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("gloo", False),
        ("cpu:gloo,cuda:nccl", False),
        ("nccl", True),
        ("cuda:nccl", True),
        ("undefined", True),
    ],
)
def test_backend_rejects_cpu_tensors(monkeypatch, backend, expected):
    _stub_dist(monkeypatch, backend)
    assert _backend_rejects_cpu_tensors() is expected


def test_no_cuda_never_rejects(monkeypatch):
    _stub_dist(monkeypatch, "nccl", cuda_available=False)
    assert _backend_rejects_cpu_tensors() is False


def test_gloo_reduces_cpu_in_place(monkeypatch):
    _stub_dist(monkeypatch, "gloo")
    seen = []
    monkeypatch.setattr(
        dist, "all_reduce", lambda buf, op=None, group=None: seen.append(buf.device)
    )
    out = _all_reduce_scalar(torch.tensor(3.0), dist.ReduceOp.SUM)
    assert [d.type for d in seen] == ["cpu"]
    assert out.item() == 3.0


@pytest.mark.parametrize("backend", ["nccl", "cuda:nccl", "undefined"])
def test_cpu_scalar_bounces_through_cuda(monkeypatch, backend):
    _stub_dist(monkeypatch, backend)
    staged = []
    orig_to = torch.Tensor.to

    def fake_to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, torch.device) and device.type == "cuda":
            staged.append(device)
            return self.clone()
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    reduced = []
    monkeypatch.setattr(
        dist, "all_reduce", lambda buf, op=None, group=None: reduced.append(buf)
    )
    out = _all_reduce_scalar(torch.tensor(3.0), dist.ReduceOp.SUM)
    assert staged and len(reduced) == 1
    assert out.item() == 3.0
