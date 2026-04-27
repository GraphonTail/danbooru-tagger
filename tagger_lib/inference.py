"""
inference.py — обёртка над DanbooruAI-моделью для Forge-расширения.

Использует функции из generate.py напрямую:
  • load_model_and_vocab()  — загружает TagTransformer + vocab
  • generate_tags()         — авторегрессивная генерация
  • load_profile()          — safety-профили из data/profiles/*.txt

Требует в tagger_lib/:
  ✓ generate.py      (из src/)
  ✓ tagger_model.py  (из src/)
В расширении:
  ✓ data/vocab_clean.json
  ✓ model/tagger_clean.pth
  ✓ data/profiles/ud_age.txt     (необязательно)
  ✓ data/profiles/ud_animals.txt (необязательно)
  ✓ data/profiles/ud_violence.txt(необязательно)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ─── пути ──────────────────────────────────────────────────────────────────
# tagger_lib/ должна быть в sys.path — danbooru_tagger.py добавляет её при старте
_HERE = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

class TaggerInference:
    """
    Загружает модель один раз, держит в памяти GPU/CPU на всё время Forge.
    Вызывай TaggerInference.get_instance(vocab_path, checkpoint_path).
    """

    _instance: Optional["TaggerInference"] = None

    # ------------------------------------------------------------------
    def __init__(self, vocab_path: str, checkpoint_path: str) -> None:
        self.vocab_path       = vocab_path
        self.checkpoint_path  = checkpoint_path

        # Всё, что нужно generate_tags(), берётся отсюда
        self._model           = None
        self._vocab: dict     = {}        # полный vocab dict (token2id, id2token, ...)
        self._token2id: Dict[str, int] = {}
        self._id2token: Dict[int, str] = {}
        self._device: str     = "cpu"

        self._profiles_cache: Dict[str, tuple] = {}  # name → (banlist, whitelist)

        self._load()

    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, vocab_path: str, checkpoint_path: str) -> "TaggerInference":
        # Перезагружаем если сменилась модель
        if cls._instance is None or cls._instance.checkpoint_path != checkpoint_path:
            if cls._instance is not None:
                print(f"[DanbooruTagger] Model changed → {Path(checkpoint_path).name}")
            cls._instance = cls(vocab_path, checkpoint_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Сбрасывает синглтон — модель перезагрузится при следующем вызове."""
        cls._instance = None

    # ══════════════════════════════════════════════════════════════════
    # Загрузка
    # ══════════════════════════════════════════════════════════════════

    def _load(self) -> None:
        try:
            from generate import load_model_and_vocab  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                f"[DanbooruTagger] Не удалось импортировать generate.py: {e}\n"
                f"  Убедись что generate.py и tagger_model.py лежат в {_HERE}/"
            ) from e

        (self._model,
         self._vocab,
         self._token2id,
         self._id2token,
         device_obj) = load_model_and_vocab(
            vocab_path      = self.vocab_path,
            checkpoint_path = self.checkpoint_path,
        )
        self._device = str(device_obj)
        print(f"[DanbooruTagger] Готово. Vocab: {len(self._token2id):,} тегов | Device: {self._device}")

    # ══════════════════════════════════════════════════════════════════
    # Safety-профили
    # ══════════════════════════════════════════════════════════════════

    def _get_profile(self, name: str) -> tuple[set, set]:
        """Загружает (banlist, whitelist) из data/profiles/<name>.txt."""
        if name in self._profiles_cache:
            return self._profiles_cache[name]

        try:
            from generate import load_profile  # noqa: PLC0415
        except ImportError:
            return set(), set()

        ban, white = load_profile(name)
        self._profiles_cache[name] = (ban, white)
        if ban or white:
            print(f"[DanbooruTagger] Профиль '{name}': {len(ban)} ban, {len(white)} white")
        return ban, white

    # ══════════════════════════════════════════════════════════════════
    # Публичный API
    # ══════════════════════════════════════════════════════════════════

    def generate(
        self,
        mode:        str,
        seed_tags:   List[str],
        count:       int   = 25,
        temperature: float = 1.0,
        spread:      float = 0.9,
        sort:        bool  = True,
        profiles:    Optional[List[str]] = None,
    ) -> List[str]:
        """
        Генерирует теги.

        mode       : 'abstract' — новые теги по затравкам (затравки в вывод не идут)
                     'addit'    — расширяет список тегов  (затравки включены)
        seed_tags  : входные теги (строки)
        count      : желаемое количество тегов в выводе
        temperature: 0.1–2.0
        spread     : nucleus p   0.1–1.0
        sort       : алфавитная сортировка
        profiles   : список safety-профилей
        """
        import torch  # noqa: PLC0415
        from generate import generate_tags  # noqa: PLC0415

        # ── Получаем ID затравок ────────────────────────────────────────
        if seed_tags:
            known    = [t for t in seed_tags if t in self._token2id]
            unknown  = [t for t in seed_tags if t not in self._token2id]
            if unknown:
                print(f"[DanbooruTagger] Теги не в словаре (пропущены): {', '.join(unknown[:10])}")
            seed_ids = [self._token2id[t] for t in known]
        else:
            seed_ids = self._random_seeds(n=3)
            print(f"[DanbooruTagger] Seed: {[self._id2token.get(i,'?') for i in seed_ids]}")

        if not seed_ids:
            seed_ids = self._random_seeds(n=3)

        # ── Собираем banlist / whitelist из профилей ────────────────────
        banlist:   set = set()
        whitelist: set = set()
        if profiles:
            for name in profiles:
                b, w = self._get_profile(name)
                banlist   |= b
                whitelist |= w

        # ── Генерация ───────────────────────────────────────────────────
        input_tags_for_addit = [self._id2token[i] for i in seed_ids
                                 if i in self._id2token] if mode == "addit" else None

        result = generate_tags(
            model      = self._model,
            vocab      = self._vocab,
            token2id   = self._token2id,
            id2token   = self._id2token,
            device     = self._device,
            seed_ids   = seed_ids,
            count      = count,
            temperature= temperature,
            spread     = spread,
            mode       = mode,
            input_tags = input_tags_for_addit,
            banlist    = banlist   if banlist   else None,
            whitelist  = whitelist if whitelist else None,
        )

        if sort:
            result = sorted(result)

        return result[:count]

    # ══════════════════════════════════════════════════════════════════
    # Утилиты
    # ══════════════════════════════════════════════════════════════════

    def _random_seeds(self, n: int = 3) -> List[int]:
        """Случайные seed-теги из топ-500 словаря."""
        special = set(self._vocab.get("special_tokens", {}).values())
        candidates = [i for i in range(min(500, len(self._token2id)))
                      if i not in special]
        return random.sample(candidates, min(n, len(candidates)))

    @property
    def vocab_size(self) -> int:
        return self._vocab.get("vocab_size", len(self._token2id))

    @property
    def device(self) -> str:
        return self._device
