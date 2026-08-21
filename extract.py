"""Emulsion Extract — ComfyUI node that recovers a grading preset from a
before/after image pair. The optimization itself lives in fitting.py.

Depends only on the stdlib and torch: importable and testable without ComfyUI.
"""

import json
import logging
import os
import re
import unicodedata
from datetime import date

try:
    from . import fitting
except ImportError:  # direct import outside the package (tests)
    import fitting

logger = logging.getLogger("ComfyUI-Emulsion")

_HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(_HERE, "presets")


def _slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s or "extracted"


def _unique_path(directory: str, slug: str) -> str:
    path = os.path.join(directory, f"{slug}.json")
    n = 2
    while os.path.exists(path):
        path = os.path.join(directory, f"{slug}_{n}.json")
        n += 1
    return path


class EmulsionExtract:
    CATEGORY = "image/postprocessing"
    FUNCTION = "extract"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_source": ("IMAGE",),
                "image_target": ("IMAGE",),
                "preset_name": ("STRING", {"default": "extracted"}),
                "iterations": ("INT", {"default": 400, "min": 50, "max": 2000}),
                "fit_vignette": ("BOOLEAN", {"default": True}),
            }
        }

    def extract(self, image_source, image_target, preset_name, iterations, fit_vignette):
        fitted, info = fitting.fit_preset(
            image_source, image_target,
            iterations=iterations, fit_vignette=fit_vignette,
        )
        preset = {
            "name": preset_name,
            "description": (
                f"Extracted by Emulsion Extract on {date.today().isoformat()} "
                f"— RMSE {info['rmse_final']:.2f}/255"
            ),
            **fitted,
        }

        os.makedirs(PRESETS_DIR, exist_ok=True)
        path = _unique_path(PRESETS_DIR, _slugify(preset_name))
        text = json.dumps(preset, ensure_ascii=False, indent=2)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

        print(
            f"[Emulsion Extract] Initial RMSE {info['rmse_init']:.2f}/255 "
            f"-> final {info['rmse_final']:.2f}/255 "
            f"({info['iterations_run']} iterations)"
        )
        print(f"[Emulsion Extract] preset written: {path}")
        return (text,)
