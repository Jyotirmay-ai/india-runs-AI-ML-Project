"""Centralized path configuration - single source of truth"""
from pathlib import Path
import os
import warnings

# Project root = folder containing main.py (auto-detected)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ── Input Paths ─────────────────────────────────────────────────────
DATASET_DIR = PROJECT_ROOT.parent / "dataset" / "dataset_provided"
CANDIDATES_PATH = DATASET_DIR / "candidates.jsonl"
JD_PATH = PROJECT_ROOT / "jd_text.txt"

# Allow env override for CI/CD (optional)
if os.getenv("DATASET_PATH"):
    CANDIDATES_PATH = Path(os.getenv("DATASET_PATH"))

# ── Cache (shared across runs) ──────────────────────────────────────
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Progress checkpoint file (persists across runs for resume capability)
PROGRESS_FILE = CACHE_DIR / "ranking_progress.jsonl"

# ── Output (outside project - intentional) ──────────────────────────
RESULT_BASE_DIR = Path(os.getenv("RESULT_BASE_DIR", PROJECT_ROOT.parent / "Result"))
RESULT_BASE_DIR.mkdir(exist_ok=True)

# ── Backward Compatibility Fallbacks ────────────────────────────────
_LEGACY_DATASET = Path(r"D:\software Projects\india runs Ai Project\dataset\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl")
_LEGACY_JD = PROJECT_ROOT / "jd_text.txt"  # Same as new
_LEGACY_CACHE = PROJECT_ROOT / "cache"       # Same as new
_LEGACY_RESULT = Path(r"D:\software Projects\india runs Ai Project\Result")

def validate_setup():
    """Call once at startup from main.py"""
    missing = []
    if not CANDIDATES_PATH.exists():
        if _LEGACY_DATASET.exists():
            warnings.warn(f"Using legacy dataset path: {_LEGACY_DATASET}")
            return  # Allow legacy
        missing.append(f"Dataset not found: {CANDIDATES_PATH}")
    if not JD_PATH.exists():
        missing.append(f"JD not found: {JD_PATH}")
    if missing:
        raise FileNotFoundError("\n".join(missing) + 
            "\n\nExpected structure:\n"
            "  project_root/\n"
            "  ├── dataset/\n"
            "  │   └── dataset_provided/\n"
            "  │       └── candidates.jsonl\n"
            "  ├── project/\n"
            "  │   ├── main.py\n"
            "  │   └── jd_text.txt\n"
            "  └── Result/ (created automatically)")

# Export commonly used paths
__all__ = [
    "PROJECT_ROOT", "CANDIDATES_PATH", "JD_PATH", 
    "CACHE_DIR", "RESULT_BASE_DIR", "validate_setup", "PROGRESS_FILE"
]