from __future__ import annotations

import ctypes
import logging
import os
from typing import Any, Callable, Iterable


LOGGER = logging.getLogger(__name__)
EGL_CUDA_DEVICE_NV = 0x323A
_PATCH_MARKER = "_progressflip_cuda_affinity_patch"


def unique_device_for_cuda_ordinal(
    devices: Iterable[Any],
    ordinal: int,
    resolver: Callable[[Any], int | None],
) -> Any:
    matches = [device for device in devices if resolver(device) == int(ordinal)]
    if len(matches) != 1:
        raise RuntimeError(
            "Could not map the requested CUDA-visible ordinal to exactly one EGL device: "
            f"ordinal={ordinal}, matches={len(matches)}. "
            "Use OSMesa or inspect the node's NVIDIA EGL installation."
        )
    return matches[0]


def install_cuda_visible_egl_affinity() -> dict[str, Any]:
    """Make robosuite select EGL by CUDA-visible logical ordinal.

    robosuite 1.4.x can treat the text in CUDA_VISIBLE_DEVICES as an index into
    EGL's independently ordered device list. This runtime-only patch asks the
    NVIDIA EGL extension for each device's CUDA ordinal and selects the device
    matching the process-local ``render_gpu_device_id`` (normally logical 0).
    It does not modify site-packages.
    """

    if os.environ.get("MUJOCO_GL", "").lower() != "egl":
        return {"active": False, "reason": "MUJOCO_GL is not egl"}

    from mujoco.egl import egl_ext as EGL
    from OpenGL import error
    from robosuite.renderers.context import egl_context

    if getattr(egl_context, _PATCH_MARKER, False):
        return {"active": True, "reason": "already installed"}
    if getattr(egl_context, "EGL_DISPLAY", None) is not None:
        raise RuntimeError(
            "ProgressFlip EGL affinity patch must run before the first EGL context is created"
        )

    address = EGL.eglGetProcAddress("eglQueryDeviceAttribEXT")
    if not address:
        raise RuntimeError(
            "NVIDIA EGL extension eglQueryDeviceAttribEXT is unavailable. "
            "Set MUJOCO_GL=osmesa as a CPU-rendering fallback."
        )
    egl_attrib = ctypes.c_ssize_t
    query_type = ctypes.CFUNCTYPE(
        EGL.EGLBoolean,
        EGL.EGLDeviceEXT,
        EGL.EGLint,
        ctypes.POINTER(egl_attrib),
    )
    query_device_attrib = query_type(address)

    def cuda_ordinal(device: Any) -> int | None:
        value = egl_attrib()
        success = query_device_attrib(
            device,
            EGL_CUDA_DEVICE_NV,
            ctypes.byref(value),
        )
        return int(value.value) if success == EGL.EGL_TRUE else None

    def create_initialized_egl_device_display(device_id: int = 0):
        all_devices = EGL.eglQueryDevicesEXT()
        logical_ordinal = 0 if int(device_id) == -1 else int(device_id)
        device = unique_device_for_cuda_ordinal(
            all_devices,
            logical_ordinal,
            cuda_ordinal,
        )
        display = EGL.eglGetPlatformDisplayEXT(
            EGL.EGL_PLATFORM_DEVICE_EXT,
            device,
            None,
        )
        if display == EGL.EGL_NO_DISPLAY or EGL.eglGetError() != EGL.EGL_SUCCESS:
            return EGL.EGL_NO_DISPLAY
        try:
            initialized = EGL.eglInitialize(display, None, None)
        except error.GLError:
            return EGL.EGL_NO_DISPLAY
        if initialized == EGL.EGL_TRUE and EGL.eglGetError() == EGL.EGL_SUCCESS:
            return display
        return EGL.EGL_NO_DISPLAY

    egl_context.create_initialized_egl_device_display = create_initialized_egl_device_display
    setattr(egl_context, _PATCH_MARKER, True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    LOGGER.info(
        "Installed CUDA-visible EGL affinity patch (CUDA_VISIBLE_DEVICES=%s)",
        visible,
    )
    return {
        "active": True,
        "reason": "installed",
        "cuda_visible_devices": visible,
    }
