"""
Shared control-type mappings and profile threshold helpers.
"""

from typing import Dict, Tuple


CONTROL_TYPES = ("canny", "openpose", "depth", "bbox")

# Per-control score threshold keys in scoring.profiles.<profile>.
PROFILE_THRESHOLD_KEYS: Dict[str, Tuple[str, str]] = {
    "canny": ("canny_auto_accept", "canny_auto_reject"),
    "openpose": ("pose_auto_accept", "pose_auto_reject"),
    "depth": ("depth_auto_accept", "depth_auto_reject"),
    "bbox": ("bbox_auto_accept", "bbox_auto_reject"),
}

# Saved control image suffix for each control type.
OUTPUT_SUFFIXES: Dict[str, str] = {
    "canny": "_canny.png",
    "openpose": "_openpose.png",
    "depth": "_depth.png",
    "bbox": "_bbox.png",
}

# JSONA / output-folder mapping for each control type.
OUTPUT_FOLDERS: Dict[str, str] = {
    "canny": "canny",
    "openpose": "pose",
    "depth": "depth",
    "bbox": "bbox",
}


def threshold_keys_for(control_type: str) -> Tuple[str, str]:
    return PROFILE_THRESHOLD_KEYS.get(str(control_type or "").strip(), ("auto_accept", "auto_reject"))


def profile_thresholds_for(
    profile: Dict,
    control_type: str,
    default_accept: float = 55.0,
    default_reject: float = 40.0,
) -> Tuple[float, float]:
    if not isinstance(profile, dict):
        profile = {}

    accept_key, reject_key = threshold_keys_for(control_type)
    accept_value = profile.get(accept_key, profile.get("auto_accept", default_accept))
    reject_value = profile.get(reject_key, profile.get("auto_reject", default_reject))

    try:
        accept = float(accept_value)
    except Exception:
        accept = float(default_accept)
    try:
        reject = float(reject_value)
    except Exception:
        reject = float(default_reject)
    return accept, reject


def output_suffix_for(control_type: str) -> str:
    key = str(control_type or "").strip()
    return OUTPUT_SUFFIXES.get(key, f"_{key}.png")


def output_folder_for(control_type: str) -> str:
    key = str(control_type or "").strip()
    if not key:
        return "unknown"
    return OUTPUT_FOLDERS.get(key, key)
