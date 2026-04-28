"""
inference.py — wrapper around the DanbooruAI model for the Forge extension.

Uses functions from generate.py directly:
  • load_model_and_vocab()  — loads TagTransformer + vocab
  • generate_tags()         — autoregressive generation
  • load_profile()          — safety profiles from data/profiles/*.txt

Requires in tagger_lib/:
  ✓ generate.py      (from src/)
  ✓ tagger_model.py  (from src/)
In the extension root:
  ✓ data/vocab_clean.json
  ✓ model/tagger_clean.pth
  ✓ data/profiles/*.txt     (optional)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# tagger_lib/ must be in sys.path — danbooru_tagger.py adds it at startup
_HERE = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

class TaggerInference:
    """
    Loads the model once and keeps it in GPU/CPU memory for the Forge session.
    Use TaggerInference.get_instance(vocab_path, checkpoint_path).
    """

    _instance: Optional["TaggerInference"] = None

    # ------------------------------------------------------------------
    def __init__(self, vocab_path: str, checkpoint_path: str) -> None:
        self.vocab_path       = vocab_path
        self.checkpoint_path  = checkpoint_path

        self._model           = None
        self._vocab: dict     = {}
        self._token2id: Dict[str, int] = {}
        self._id2token: Dict[int, str] = {}
        self._device: str     = "cpu"

        self._profiles_cache: Dict[str, tuple] = {}   # name → (banlist, whitelist)

        self._load()

    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, vocab_path: str, checkpoint_path: str) -> "TaggerInference":
        # Reload if the model file changed
        if cls._instance is None or cls._instance.checkpoint_path != checkpoint_path:
            if cls._instance is not None:
                print(f"[DanbooruTagger] Model changed → {Path(checkpoint_path).name}")
            cls._instance = cls(vocab_path, checkpoint_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton — model reloads on next call."""
        cls._instance = None

    # ══════════════════════════════════════════════════════════════════
    # Loading
    # ══════════════════════════════════════════════════════════════════

    def _load(self) -> None:
        try:
            from generate import load_model_and_vocab  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                f"[DanbooruTagger] Could not import generate.py: {e}\n"
                f"  Make sure generate.py and tagger_model.py are in {_HERE}/"
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
        print(f"[DanbooruTagger] Ready. Vocab: {len(self._token2id):,} tags | Device: {self._device}")

    # ══════════════════════════════════════════════════════════════════
    # Safety profiles
    # ══════════════════════════════════════════════════════════════════

    def _get_profile(self, name: str) -> tuple[set, set]:
        """Load (banlist, whitelist) from data/profiles/<name>.txt."""
        if name in self._profiles_cache:
            return self._profiles_cache[name]

        try:
            from generate import load_profile  # noqa: PLC0415
        except ImportError:
            return set(), set()

        ban, white = load_profile(name)
        self._profiles_cache[name] = (ban, white)
        if ban or white:
            print(f"[DanbooruTagger] Profile '{name}': {len(ban)} ban, {len(white)} white")
        return ban, white

    # ══════════════════════════════════════════════════════════════════
    # Public API
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
        Generate tags.

        mode       : 'abstract' — new tags conditioned on seeds (seeds not in output)
                     'addit'    — extends seed list (seeds included in output)
        seed_tags  : input seed tags (strings)
        count      : desired number of output tags
        temperature: sampling temperature (0.1–50)
        spread     : nucleus-p diversity (0.1–50)
        sort       : alphabetical sort of output
        profiles   : list of safety-profile names to apply
        """
        import torch  # noqa: PLC0415
        from generate import generate_tags  # noqa: PLC0415

        # ── Resolve seed IDs ────────────────────────────────────────────
        if seed_tags:
            known   = [t for t in seed_tags if t in self._token2id]
            unknown = [t for t in seed_tags if t not in self._token2id]
            if unknown:
                print(f"[DanbooruTagger] Unknown seed tags (skipped): {', '.join(unknown[:10])}")
            seed_ids = [self._token2id[t] for t in known]
        else:
            seed_ids = self._random_seeds(n=3)
            print(f"[DanbooruTagger] Random seeds: {[self._id2token.get(i, '?') for i in seed_ids]}")

        if not seed_ids:
            seed_ids = self._random_seeds(n=3)

        # ── Merge banlist / whitelist from profiles ─────────────────────
        banlist:   set = set()
        whitelist: set = set()
        if profiles:
            for name in profiles:
                b, w = self._get_profile(name)
                banlist   |= b
                whitelist |= w

        # ── Generate ────────────────────────────────────────────────────
        input_tags_for_addit = (
            [self._id2token[i] for i in seed_ids if i in self._id2token]
            if mode == "addit" else None
        )

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
    # Utilities
    # ══════════════════════════════════════════════════════════════════

    def _random_seeds(self, n: int = 3) -> List[int]:
        """Pick random seed tags from the top-500 vocabulary entries."""
        special    = set(self._vocab.get("special_tokens", {}).values())
        candidates = [i for i in range(min(500, len(self._token2id)))
                      if i not in special]
        return random.sample(candidates, min(n, len(candidates)))

    @property
    def vocab_size(self) -> int:
        return self._vocab.get("vocab_size", len(self._token2id))

    @property
    def device(self) -> str:
        return self._device
