from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GeomGroups:
    robot: frozenset[int]
    eef: frozenset[int]
    arm: frozenset[int]

    @classmethod
    def from_ids(cls, robot: Iterable[int], eef: Iterable[int]) -> "GeomGroups":
        robot_set = frozenset(int(item) for item in robot)
        eef_set = frozenset(int(item) for item in eef) & robot_set
        return cls(robot=robot_set, eef=eef_set, arm=robot_set - eef_set)


@dataclass(frozen=True)
class Capture:
    rgb: np.ndarray
    segmentation: np.ndarray


@dataclass(frozen=True)
class CompositeBundle:
    true: np.ndarray
    true_recomposed: np.ndarray
    full_old: np.ndarray
    robot_old: np.ndarray
    nonrobot_old: np.ndarray
    eef_old: np.ndarray
    arm_old: np.ndarray
    robot_mask: np.ndarray
    robot_random: np.ndarray | None = None
    robot_phase25: np.ndarray | None = None
    robot_phase50: np.ndarray | None = None
    robot_phase75: np.ndarray | None = None

    def get(self, name: str) -> np.ndarray:
        value = getattr(self, name)
        if value is None:
            raise KeyError(f"Composite {name!r} was not generated")
        return value


def geom_mask(segmentation: np.ndarray, geom_ids: Iterable[int]) -> np.ndarray:
    seg = np.asarray(segmentation)
    if seg.ndim != 3 or seg.shape[-1] < 2:
        raise ValueError(f"Expected segmentation [H,W,2+], got {seg.shape}")
    ids = np.asarray(sorted(set(int(item) for item in geom_ids)), dtype=np.int64)
    if ids.size == 0:
        return np.zeros(seg.shape[:2], dtype=bool)
    try:
        import mujoco

        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    except Exception:
        geom_type = 5
    return (seg[..., 0].astype(np.int64) == geom_type) & np.isin(
        seg[..., 1].astype(np.int64), ids
    )


def expand_mask(mask: np.ndarray, dilation_px: int = 2, feather_px: float = 0.0) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if dilation_px > 0:
        binary = ndimage.binary_dilation(binary, iterations=int(dilation_px))
    alpha = binary.astype(np.float32)
    if feather_px > 0:
        alpha = ndimage.gaussian_filter(alpha, sigma=float(feather_px))
        maximum = float(alpha.max())
        if maximum > 0:
            alpha /= maximum
    return np.clip(alpha, 0.0, 1.0)


def inpaint_nearest(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    binary = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB [H,W,3], got {rgb.shape}")
    if binary.shape != rgb.shape[:2]:
        raise ValueError("Image and mask shapes do not match")
    if not binary.any():
        return rgb.copy()
    if binary.all():
        raise ValueError("Cannot inpaint an all-true mask")
    nearest = ndimage.distance_transform_edt(binary, return_distances=False, return_indices=True)
    output = rgb.copy()
    output[binary] = rgb[nearest[0][binary], nearest[1][binary]]
    return output


def paste(background: np.ndarray, foreground: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    bg = np.asarray(background, dtype=np.float32)
    fg = np.asarray(foreground, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    if bg.shape != fg.shape or bg.shape[:2] != a.shape:
        raise ValueError("Background, foreground, and alpha shapes are incompatible")
    mixed = bg * (1.0 - a[..., None]) + fg * a[..., None]
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _robot_composite(
    current: Capture,
    donor: Capture,
    current_ids: Iterable[int],
    donor_ids: Iterable[int],
    dilation_px: int,
    feather_px: float,
) -> np.ndarray:
    current_mask = geom_mask(current.segmentation, current_ids)
    donor_mask = geom_mask(donor.segmentation, donor_ids)
    background = inpaint_nearest(
        current.rgb, ndimage.binary_dilation(current_mask, iterations=dilation_px)
    )
    alpha = expand_mask(donor_mask, dilation_px=dilation_px, feather_px=feather_px)
    return paste(background, donor.rgb, alpha)


def build_composites(
    current: Capture,
    old: Capture,
    groups: GeomGroups,
    *,
    random_capture: Capture | None = None,
    phase_captures: dict[int, Capture] | None = None,
    dilation_px: int = 2,
    feather_px: float = 0.0,
) -> CompositeBundle:
    current_robot_mask = geom_mask(current.segmentation, groups.robot)
    old_robot_mask = geom_mask(old.segmentation, groups.robot)
    current_bg = inpaint_nearest(
        current.rgb, ndimage.binary_dilation(current_robot_mask, iterations=dilation_px)
    )
    old_bg = inpaint_nearest(
        old.rgb, ndimage.binary_dilation(old_robot_mask, iterations=dilation_px)
    )
    true_recomposed = paste(
        current_bg,
        current.rgb,
        expand_mask(current_robot_mask, dilation_px=dilation_px, feather_px=feather_px),
    )
    robot_old = _robot_composite(
        current, old, groups.robot, groups.robot, dilation_px, feather_px
    )
    nonrobot_old = paste(
        old_bg,
        current.rgb,
        expand_mask(current_robot_mask, dilation_px=dilation_px, feather_px=feather_px),
    )
    current_arm = _robot_composite(
        current, current, groups.arm, groups.arm, dilation_px, feather_px
    )
    eef_old = _robot_composite(
        Capture(current_arm, current.segmentation),
        old,
        groups.eef,
        groups.eef,
        dilation_px,
        feather_px,
    )
    current_eef = _robot_composite(
        current, current, groups.eef, groups.eef, dilation_px, feather_px
    )
    arm_old = _robot_composite(
        Capture(current_eef, current.segmentation),
        old,
        groups.arm,
        groups.arm,
        dilation_px,
        feather_px,
    )
    random_image = None
    if random_capture is not None:
        random_image = _robot_composite(
            current, random_capture, groups.robot, groups.robot, dilation_px, feather_px
        )
    phase_images: dict[int, np.ndarray | None] = {25: None, 50: None, 75: None}
    for fraction, capture in (phase_captures or {}).items():
        if fraction in phase_images:
            phase_images[fraction] = _robot_composite(
                current, capture, groups.robot, groups.robot, dilation_px, feather_px
            )
    return CompositeBundle(
        true=current.rgb.copy(),
        true_recomposed=true_recomposed,
        full_old=old.rgb.copy(),
        robot_old=robot_old,
        nonrobot_old=nonrobot_old,
        eef_old=eef_old,
        arm_old=arm_old,
        robot_mask=current_bg,
        robot_random=random_image,
        robot_phase25=phase_images[25],
        robot_phase50=phase_images[50],
        robot_phase75=phase_images[75],
    )


def image_mae(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        return float("nan")
    return float(np.mean(np.abs(aa - bb)))


def changed_fraction(a: np.ndarray, b: np.ndarray, threshold: float = 5.0) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        return float("nan")
    delta = np.max(np.abs(aa - bb), axis=-1)
    return float(np.mean(delta > threshold))
