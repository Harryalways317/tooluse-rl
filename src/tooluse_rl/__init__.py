"""tooluse-rl: RL post-training (GRPO) of a small open model on tau2-bench retail.

Importing this package has one deliberate side effect: it points tau2-bench at
the pinned benchmark data under ``data/tau2-bench`` (see :mod:`tooluse_rl.data`)
unless ``TAU2_DATA_DIR`` is already set, because tau2 resolves its data
directory at import time.

The package root is also the ``verifiers`` plugin entry point. ``verifiers``
resolves the taskset id ``tooluse-rl`` to this module and looks in ``__all__``
for a ``Taskset`` subclass and an ``Env`` subclass; both are imported lazily so
that ``import tooluse_rl`` stays cheap.
"""

from __future__ import annotations

from tooluse_rl.data import ensure_tau2_data_dir

__version__ = "0.1.0"

# verifiers reads this list: exactly one Taskset subclass and one Env subclass.
__all__ = ["Tau2Taskset", "Tau2UserSimEnv", "__version__"]

ensure_tau2_data_dir()


def __getattr__(name: str):
    if name in {"Tau2Taskset", "Tau2UserSimEnv"}:
        from tooluse_rl import vf

        return getattr(vf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
