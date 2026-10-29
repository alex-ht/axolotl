"""Feature detection for the installed axolotl-ringmaster."""

from __future__ import annotations


def ringmaster_supports_ulysses_flex() -> bool:
    """True iff the installed ringmaster builds a causal / same-document /
    sliding-window BlockMask for the Ulysses flex_attention inner (exact forward and
    backward). Older releases ran flex unmasked or fell back to a dense SDPA mask."""
    try:
        from ringmaster.strategies.ulysses import SUPPORTS_FLEX_BLOCK_MASK
    except ImportError:
        return False
    return bool(SUPPORTS_FLEX_BLOCK_MASK)
