"""
install.py — запускается ForgeNeo автоматически при загрузке расширения.
Проверяет наличие зависимостей, создаёт нужные папки и подсказывает
пользователю что нужно скачать.
"""
import importlib
import subprocess
import sys
from pathlib import Path

# ─── папки ─────────────────────────────────────────────────────────────────
EXT_DIR      = Path(__file__).resolve().parent
MODEL_DIR    = EXT_DIR / "model"
DATA_DIR     = EXT_DIR / "data"
PROFILES_DIR = DATA_DIR / "profiles"

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PROFILES_DIR.mkdir(exist_ok=True)

# ─── подсказка если модель не скачана ──────────────────────────────────────
_model_missing = not any(MODEL_DIR.glob("*.pth"))
_vocab_missing = not (DATA_DIR / "vocab_clean.json").exists()

if _model_missing or _vocab_missing:
    print("=" * 60)
    print("[DanbooruTagger] ⚠  Model files not found!")
    print("  Download from: https://github.com/GraphonTail/danbooru-tagger/releases")
    if _model_missing:
        print(f"  Place  tagger_clean.pth  →  {MODEL_DIR}")
    if _vocab_missing:
        print(f"  Place  vocab_clean.json  →  {DATA_DIR}")
    print("=" * 60)
else:
    print("[DanbooruTagger] Model files OK.")


# ─── зависимости ───────────────────────────────────────────────────────────
def install(package: str, import_name: str | None = None) -> None:
    name = import_name or package
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"[DanbooruTagger] Installing {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", package]
        )
        print(f"[DanbooruTagger] {package} installed.")


# torch уже есть в Forge — не переустанавливаем
# Остальные зависимости тоже входят в стандартный Forge-стек,
# но на случай нестандартной среды проверяем:
install("numpy")
install("tqdm")

print("[DanbooruTagger] Dependencies OK.")
