"""
Fusion score pre-filter for original images.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class FusionScoreFilter:
    REQUIRED_MODULES = {
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "open_clip": "open-clip-torch",
        "timm": "timm",
        "safetensors": "safetensors",
        "yaml": "PyYAML",
        "tqdm": "tqdm",
        "PIL": "Pillow",
        "einops": "einops",
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = self.normalize_config(config)
        self._lock = threading.RLock()
        self._probe_cache: Optional[dict] = None
        self._runtime = None
        self._runtime_module = None
        self._runtime_error = ""

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def default_config(cls) -> dict:
        return {
            "enabled": False,
            "checkpoint_path": "./models/score/5kdataset.safetensors",
            "cache_root": "./models/score",
            "device": "auto",
            "min_aesthetic_score": 2.5,
            "require_in_domain": True,
            "min_in_domain_prob": 0.5,
        }

    @classmethod
    def normalize_config(cls, config: Optional[dict]) -> dict:
        merged = dict(cls.default_config())
        if isinstance(config, dict):
            merged.update(config)

        merged["enabled"] = bool(merged.get("enabled", False))
        merged["checkpoint_path"] = str(
            merged.get("checkpoint_path") or cls.default_config()["checkpoint_path"]
        ).strip()
        merged["cache_root"] = str(
            merged.get("cache_root") or cls.default_config()["cache_root"]
        ).strip()

        device = str(merged.get("device") or "auto").strip().lower()
        if device not in {"auto", "cpu", "cuda"}:
            device = "auto"
        merged["device"] = device

        try:
            merged["min_aesthetic_score"] = float(
                merged.get("min_aesthetic_score", 2.5)
            )
        except Exception:
            merged["min_aesthetic_score"] = 2.5
        merged["min_aesthetic_score"] = max(
            0.0, min(5.0, merged["min_aesthetic_score"])
        )

        merged["require_in_domain"] = bool(merged.get("require_in_domain", True))
        try:
            merged["min_in_domain_prob"] = float(
                merged.get("min_in_domain_prob", 0.5)
            )
        except Exception:
            merged["min_in_domain_prob"] = 0.5
        merged["min_in_domain_prob"] = max(0.0, min(1.0, merged["min_in_domain_prob"]))
        return merged

    def is_enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path).strip()).expanduser()
        if not path.is_absolute():
            path = (self.repo_root / path).resolve()
        return path

    @classmethod
    def _format_path_message(cls, label: str, path: Path) -> str:
        return f"{label}: {path}"

    @classmethod
    def _read_checkpoint_metadata(cls, checkpoint_path: Path) -> dict:
        from safetensors import safe_open

        with safe_open(str(checkpoint_path), framework="np") as handle:
            metadata = handle.metadata() or {}

        config_json = metadata.get("config_json")
        if not config_json:
            raise ValueError(
                f"Checkpoint metadata missing config_json: {checkpoint_path}"
            )

        config = json.loads(config_json)
        return {
            "config": config,
            "input_dim": int(metadata.get("input_dim", 0) or 0),
            "hidden_dims": json.loads(
                metadata.get("hidden_dims_json")
                or metadata.get("hidden_dims", "[]")
                or "[]"
            ),
            "dropout": float(metadata.get("dropout", 0.2) or 0.2),
        }

    @classmethod
    def format_probe_message(cls, probe: dict) -> str:
        if not probe.get("enabled", False):
            return "未启用"

        parts = []
        parts.append("已就绪" if probe.get("available", False) else "不可用")

        missing_deps = probe.get("missing_dependencies", []) or []
        if missing_deps:
            parts.append("缺少依赖: " + ", ".join(missing_deps))

        missing_files = probe.get("missing_files", []) or []
        if missing_files:
            parts.append("缺少文件: " + "；".join(missing_files))

        warnings = probe.get("warnings", []) or []
        if warnings:
            parts.append("提示: " + "；".join(warnings))

        runtime_error = str(probe.get("runtime_error", "") or "").strip()
        if runtime_error:
            parts.append("运行时错误: " + runtime_error)

        return "；".join(parts)

    @classmethod
    def probe_environment(cls, config: Optional[dict] = None) -> dict:
        cfg = cls.normalize_config(config)
        repo_root = Path(__file__).resolve().parents[2]

        def _resolve(raw: str) -> Path:
            path = Path(str(raw).strip()).expanduser()
            if not path.is_absolute():
                path = (repo_root / path).resolve()
            return path

        probe = {
            "enabled": bool(cfg.get("enabled", False)),
            "available": False,
            "checkpoint_path": str(_resolve(cfg.get("checkpoint_path", ""))),
            "cache_root": str(_resolve(cfg.get("cache_root", ""))),
            "missing_dependencies": [],
            "missing_files": [],
            "warnings": [],
            "runtime_error": "",
            "metadata": {},
        }

        if not probe["enabled"]:
            probe["message"] = cls.format_probe_message(probe)
            return probe

        for module_name, package_name in cls.REQUIRED_MODULES.items():
            if importlib.util.find_spec(module_name) is None:
                probe["missing_dependencies"].append(package_name)

        checkpoint_path = Path(probe["checkpoint_path"])
        cache_root = Path(probe["cache_root"])
        if not checkpoint_path.exists():
            probe["missing_files"].append(
                cls._format_path_message("评分 checkpoint", checkpoint_path)
            )

        metadata = {}
        if checkpoint_path.exists() and importlib.util.find_spec("safetensors") is not None:
            try:
                metadata = cls._read_checkpoint_metadata(checkpoint_path)
                probe["metadata"] = metadata
            except Exception as exc:
                probe["missing_files"].append(f"checkpoint 元数据读取失败: {exc}")

        models_cfg = (metadata.get("config") or {}).get("models", {})
        include_waifu_score = bool(models_cfg.get("include_waifu_score", True))
        waifu_head = cache_root / "waifu-scorer-v3" / "model.safetensors"
        if include_waifu_score and not waifu_head.exists():
            probe["missing_files"].append(
                cls._format_path_message("waifu scorer head", waifu_head)
            )

        jtp_repo = cache_root / "repos" / "RedRocket__JTP-3"
        if not (jtp_repo / "model.py").exists() or not (
            jtp_repo / "models" / "jtp-3-hydra.safetensors"
        ).exists():
            probe["warnings"].append(
                f"未检测到本地 JTP-3 缓存: {jtp_repo}。首次运行可能需要联网下载。"
            )

        probe["available"] = not probe["missing_dependencies"] and not probe["missing_files"]
        probe["message"] = cls.format_probe_message(probe)
        return probe

    def get_probe(self, force_refresh: bool = False) -> dict:
        with self._lock:
            if force_refresh or self._probe_cache is None:
                self._probe_cache = self.probe_environment(self.config)
                if self._runtime_error:
                    self._probe_cache["runtime_error"] = self._runtime_error
                    self._probe_cache["message"] = self.format_probe_message(
                        self._probe_cache
                    )
            return dict(self._probe_cache)

    def _load_runtime_module(self):
        runtime_path = self.repo_root / "ref" / "batch" / "runtime" / "batch_infer.py"
        if not runtime_path.exists():
            raise FileNotFoundError(f"评分运行时不存在: {runtime_path}")

        module_name = "_controlnet_gui_fusion_batch_infer"
        if module_name in sys.modules:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, runtime_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载评分运行时: {runtime_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def ensure_runtime_ready(self) -> tuple[bool, str]:
        if not self.is_enabled():
            return True, ""

        with self._lock:
            if self._runtime is not None:
                return True, ""

            probe = self.get_probe(force_refresh=True)
            if not probe.get("available", False):
                return False, str(probe.get("message", "评分环境不可用"))

            checkpoint_path = self._resolve_path(self.config.get("checkpoint_path", ""))
            cache_root = self._resolve_path(self.config.get("cache_root", ""))
            env_patch = {"FUSION_MODEL_CACHE_ROOT": str(cache_root)}
            waifu_head = cache_root / "waifu-scorer-v3" / "model.safetensors"
            if waifu_head.exists():
                env_patch["FUSION_WAIFU_V3_HEAD_PATH"] = str(waifu_head)

            runtime_module = self._load_runtime_module()
            previous_env = {key: os.environ.get(key) for key in env_patch}
            try:
                for key, value in env_patch.items():
                    os.environ[key] = value
                self._runtime = runtime_module._load_runtime(
                    checkpoint_path, self.config.get("device", "auto")
                )
                self._runtime_module = runtime_module
                self._runtime_error = ""
                self._probe_cache = None
                return True, ""
            except Exception as exc:
                self._runtime = None
                self._runtime_module = None
                self._runtime_error = str(exc)
                self._probe_cache = None
                return False, f"评分模型初始化失败: {exc}"
            finally:
                for key, old_value in previous_env.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value

    def evaluate(self, image) -> Dict[str, Any]:
        result = {
            "enabled": self.is_enabled(),
            "available": False,
            "device": None,
            "aesthetic": None,
            "composition": None,
            "color": None,
            "sexual": None,
            "in_domain_prob": None,
            "in_domain_pred": None,
            "passed": True,
            "should_reject": False,
            "reason": "",
        }
        if not self.is_enabled():
            return result

        ok, error = self.ensure_runtime_ready()
        if not ok:
            raise RuntimeError(error)

        import torch

        runtime = self._runtime
        pil_image = image.convert("RGB") if getattr(image, "mode", "") != "RGB" else image
        with torch.no_grad():
            f1 = runtime["jtp"]([pil_image])
            f2 = runtime["waifu"]([pil_image])
            reg_pred, cls_logit = runtime["head"](torch.cat([f1, f2], dim=-1))
            reg_row = reg_pred.detach().cpu().tolist()[0]
            cls_prob = None
            if runtime.get("has_cls_head", False):
                cls_prob = float(torch.sigmoid(cls_logit).detach().cpu().view(-1)[0].item())

        aesthetic = float(reg_row[0]) if len(reg_row) > 0 else None
        composition = float(reg_row[1]) if len(reg_row) > 1 else None
        color = float(reg_row[2]) if len(reg_row) > 2 else None
        sexual = float(reg_row[3]) if len(reg_row) > 3 else None
        in_domain_prob = None if cls_prob is None else float(cls_prob)
        threshold = float(self.config.get("min_in_domain_prob", 0.5))
        in_domain_pred = None if in_domain_prob is None else (1 if in_domain_prob >= threshold else 0)

        passed = True
        reasons = []
        min_aesthetic = float(self.config.get("min_aesthetic_score", 2.5))
        if aesthetic is None:
            passed = False
            reasons.append("评分结果缺少 aesthetic 分数")
        elif aesthetic < min_aesthetic:
            passed = False
            reasons.append(f"aesthetic {aesthetic:.3f} < {min_aesthetic:.3f}")

        if bool(self.config.get("require_in_domain", True)):
            if in_domain_prob is None:
                passed = False
                reasons.append("目标域概率不可用")
            elif in_domain_prob < threshold:
                passed = False
                reasons.append(f"in_domain_prob {in_domain_prob:.3f} < {threshold:.3f}")

        result.update(
            {
                "available": True,
                "device": runtime.get("device", self.config.get("device", "auto")),
                "aesthetic": aesthetic,
                "composition": composition,
                "color": color,
                "sexual": sexual,
                "in_domain_prob": in_domain_prob,
                "in_domain_pred": in_domain_pred,
                "passed": passed,
                "should_reject": not passed,
                "reason": "; ".join(reasons),
            }
        )
        return result

