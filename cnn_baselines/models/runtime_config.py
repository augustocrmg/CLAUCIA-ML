"""Shared TensorFlow runtime configuration for all model scripts in this repo.

Why this file exists
====================

Every model script in ``models/<arch>/<arch>_mlflow.py`` used to have its own
inline GPU-init block::

    print("GPU Available: ", tf.config.list_physical_devices('GPU'))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print('GPU memory growth enabled')
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print('Mixed precision enabled')
        except RuntimeError as e:
            print(e)

That block silently fails on **NVIDIA Blackwell GPUs** (compute capability
``12.x``: RTX PRO 6000 Blackwell, RTX 5080, RTX 5090, ...). On those cards
every publicly shipping ``tensorflow`` wheel through TF 2.20 has broken
``sm_120`` PTX. The device is visible and small ops succeed, but real graph
execution crashes during ``model_construction`` or ``model.fit`` with::

    'cuModuleLoadData(&module, data)' failed with 'CUDA_ERROR_INVALID_PTX'
    'cuLaunchKernel(...)' failed with 'CUDA_ERROR_INVALID_HANDLE'  [Op:Cast]

The crash is reliably triggered by the ``Cast`` ops that ``mixed_float16``
inserts (e.g. inside ``layers.Normalization`` of every Keras EfficientNet).

See the upstream tracking issue:
https://github.com/tensorflow/tensorflow/issues/112272

Behaviour
=========

``configure_runtime()`` honors two environment variables, in priority order:

* ``FORCE_CPU=1`` -- hide every GPU and run on CPU only (slow, but a guaranteed
  end-to-end smoke path on Blackwell).
* ``DISABLE_MIXED_PRECISION=1`` -- keep the GPU but skip ``mixed_float16``
  globally. This alone unblocks model construction on Blackwell and is enough
  for graphs that do not need string/ragged kernels (i.e. our CNN classifiers).

If neither env var is set, the helper inspects the GPU's compute capability:

* compute capability ``>= 12`` (Blackwell or newer) -> mixed precision is
  auto-disabled with a clear warning that points at the upstream bug.
* otherwise (Ampere/Ada/Hopper/...) -> ``mixed_float16`` is enabled, exactly
  like the original inline block.

In every case ``tf.config.set_soft_device_placement(True)`` is enabled so a
single op without a working GPU kernel falls back to CPU instead of taking
down the whole graph.

Usage
=====

At the top of every model script, replace the old GPU-init block with::

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runtime_config import configure_runtime
    configure_runtime()
"""

from __future__ import annotations

import os
from typing import Tuple

import tensorflow as tf


_BLACKWELL_TRACKING_ISSUE = (
    "https://github.com/tensorflow/tensorflow/issues/112272"
)


def _get_compute_capability(gpu) -> Tuple[int, int]:
    """Return ``(major, minor)`` compute capability for ``gpu``.

    Falls back to ``(0, 0)`` if TensorFlow cannot report it on this build.
    """
    try:
        details = tf.config.experimental.get_device_details(gpu)
        cc = details.get("compute_capability", (0, 0))
        if isinstance(cc, (tuple, list)) and len(cc) >= 2:
            return int(cc[0]), int(cc[1])
    except Exception as exc:  # noqa: BLE001 - defensive, must never crash here
        print(f"[runtime] could not query compute capability ({exc})")
    return 0, 0


def configure_runtime(*, prefer_mixed_precision: bool = True) -> None:
    """Configure TF to run safely on the current GPU (Blackwell-aware).

    Parameters
    ----------
    prefer_mixed_precision
        When ``True`` (default), ``mixed_float16`` is enabled on non-Blackwell
        GPUs. When ``False``, mixed precision is never enabled regardless of
        GPU. This is useful for models that do not benefit from fp16.
    """
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    disable_mp_env = os.environ.get("DISABLE_MIXED_PRECISION", "0") == "1"

    if force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            print(f"[runtime] could not hide GPU ({exc}); continuing")
        print("[runtime] FORCE_CPU=1 -> GPU disabled, mixed precision disabled")
        tf.config.set_soft_device_placement(True)
        return

    gpus = tf.config.list_physical_devices("GPU")
    print("GPU Available: ", gpus)

    if not gpus:
        print("[runtime] no GPU visible; running on CPU")
        tf.config.set_soft_device_placement(True)
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)
    print("GPU memory growth enabled")

    cc_major, cc_minor = _get_compute_capability(gpus[0])
    is_blackwell = cc_major >= 12
    if is_blackwell:
        print(
            f"[runtime] Detected GPU compute capability {cc_major}.{cc_minor} "
            "(Blackwell or newer)."
        )
        print(
            "[runtime] TF<=2.20 has a known PTX/JIT bug on sm_120 "
            f"(see {_BLACKWELL_TRACKING_ISSUE})."
        )
        print(
            "[runtime] Auto-disabling mixed_float16 to avoid Cast-op crashes "
            "during model construction."
        )
        print(
            "[runtime] If model.fit still crashes with CUDA_ERROR_INVALID_HANDLE, "
            "rerun with FORCE_CPU=1 or use a non-Blackwell GPU."
        )

    use_mp = prefer_mixed_precision and not is_blackwell and not disable_mp_env

    if use_mp:
        try:
            tf.keras.mixed_precision.set_global_policy(
                tf.keras.mixed_precision.Policy("mixed_float16")
            )
            print("Mixed precision enabled")
        except Exception as exc:  # noqa: BLE001 - never let init crash the script
            print(f"[runtime] failed to enable mixed precision: {exc}")
    else:
        try:
            tf.keras.mixed_precision.set_global_policy("float32")
        except Exception:  # noqa: BLE001
            pass
        reason = []
        if is_blackwell:
            reason.append("Blackwell")
        if disable_mp_env:
            reason.append("DISABLE_MIXED_PRECISION=1")
        if not prefer_mixed_precision:
            reason.append("prefer_mixed_precision=False")
        if not reason:
            reason.append("no GPU")
        print(f"Mixed precision DISABLED ({', '.join(reason)})")

    tf.config.set_soft_device_placement(True)


__all__ = ["configure_runtime"]
