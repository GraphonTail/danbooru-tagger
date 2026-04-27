"""
install.py — запускается ForgeNeo автоматически при загрузке расширения.
Проверяет наличие зависимостей и устанавливает недостающие.
"""
import importlib
import subprocess
import sys


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
