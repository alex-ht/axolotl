"""Tests for the flex_attention kernel_options override (H100 shared-memory OOM workaround)."""

from axolotl.monkeypatch.attention import flex_attn as fa


def test_patch_and_unpatch_flex_kernel_options():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["flex_attention"]

    assert fa.patch_flex_kernel_options({"BLOCK_M": 16, "BLOCK_N": 16}) is True
    assert ALL_ATTENTION_FUNCTIONS["flex_attention"] is not original
    assert (
        fa.patch_flex_kernel_options({"BLOCK_M": 16, "BLOCK_N": 16}) is True
    )  # idempotent

    fa.unpatch_flex_kernel_options()
    assert ALL_ATTENTION_FUNCTIONS["flex_attention"] is original


def test_wrapped_forward_injects_kernel_options(monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    captured = {}

    def fake_original(module, query, key, value, attention_mask, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(
        "transformers.integrations.flex_attention.flex_attention_forward",
        fake_original,
    )

    kernel_options = {"BLOCK_M": 16, "BLOCK_N": 16, "num_stages": 1, "num_warps": 4}
    fa.patch_flex_kernel_options(kernel_options)
    try:
        wrapped = ALL_ATTENTION_FUNCTIONS["flex_attention"]
        result = wrapped(None, None, None, None, None)
        assert result == "ok"
        assert captured["kernel_options"] == kernel_options
    finally:
        # unpatch_flex_kernel_options() would restore `fake_original` here, since that's
        # what `original` resolved to inside patch_flex_kernel_options (we monkeypatched
        # the module attribute it imports from) -- register the true original directly.
        ALL_ATTENTION_FUNCTIONS.register("flex_attention", original)


def test_wrapped_forward_does_not_override_explicit_kernel_options(monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    captured = {}

    def fake_original(module, query, key, value, attention_mask, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(
        "transformers.integrations.flex_attention.flex_attention_forward",
        fake_original,
    )

    fa.patch_flex_kernel_options({"BLOCK_M": 16, "BLOCK_N": 16})
    try:
        wrapped = ALL_ATTENTION_FUNCTIONS["flex_attention"]
        explicit = {"BLOCK_M": 64, "BLOCK_N": 64}
        wrapped(None, None, None, None, None, kernel_options=explicit)
        assert captured["kernel_options"] == explicit
    finally:
        # unpatch_flex_kernel_options() would restore `fake_original` here, since that's
        # what `original` resolved to inside patch_flex_kernel_options (we monkeypatched
        # the module attribute it imports from) -- register the true original directly.
        ALL_ATTENTION_FUNCTIONS.register("flex_attention", original)


def test_resolve_defaults_when_unset():
    assert fa.resolve_flex_kernel_options(None) == {
        "num_stages": 1,
        "bwd_num_stages": 1,
    }
    assert fa.resolve_flex_kernel_options({}) == {"num_stages": 1, "bwd_num_stages": 1}


def test_resolve_user_merged_over_defaults():
    assert fa.resolve_flex_kernel_options(
        {"BLOCK_M": 16, "BLOCK_N": 16, "num_stages": 2}
    ) == {"num_stages": 2, "bwd_num_stages": 1, "BLOCK_M": 16, "BLOCK_N": 16}


def test_resolve_none_drops_key_and_opts_out():
    assert fa.resolve_flex_kernel_options({"bwd_num_stages": None}) == {"num_stages": 1}
    assert (
        fa.resolve_flex_kernel_options({"num_stages": None, "bwd_num_stages": None})
        == {}
    )


def test_resolve_does_not_mutate_defaults():
    fa.resolve_flex_kernel_options({"num_stages": 3, "bwd_num_stages": None})
    assert fa.DEFAULT_FLEX_KERNEL_OPTIONS == {"num_stages": 1, "bwd_num_stages": 1}


def _run_patch_manager(monkeypatch, cfg):
    from types import SimpleNamespace

    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.utils.dict import DictDefault

    calls = []
    monkeypatch.setattr(fa, "patch_flex_kernel_options", calls.append)
    PatchManager._apply_flex_attn_kernel_options_patch(
        SimpleNamespace(cfg=DictDefault(cfg))
    )
    return calls


def test_patch_manager_applies_defaults_for_flex(monkeypatch):
    calls = _run_patch_manager(monkeypatch, {"attn_implementation": "flex_attention"})
    assert calls == [{"num_stages": 1, "bwd_num_stages": 1}]


def test_patch_manager_merges_user_options(monkeypatch):
    calls = _run_patch_manager(
        monkeypatch,
        {
            "attn_implementation": "flex_attention",
            "flex_attn_kernel_options": {"BLOCK_M": 16, "bwd_num_stages": 2},
        },
    )
    assert calls == [{"num_stages": 1, "bwd_num_stages": 2, "BLOCK_M": 16}]


def test_patch_manager_opt_out_skips_patch(monkeypatch):
    calls = _run_patch_manager(
        monkeypatch,
        {
            "attn_implementation": "flex_attention",
            "flex_attn_kernel_options": {"num_stages": None, "bwd_num_stages": None},
        },
    )
    assert calls == []


def test_patch_manager_skips_non_flex(monkeypatch):
    assert _run_patch_manager(monkeypatch, {"attn_implementation": "sdpa"}) == []
