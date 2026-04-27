"""
generate.py
===========
Генерация тегов в двух режимах: Абстракт и Аддитор.

АБСТРАКТ — генерирует новые теги на основе входных (входные в результат НЕ включаются):
  python src/generate.py abstract "1girl, sword"
  python src/generate.py abstract "1girl, forest, dark" --count 20 --temp 0.8 --spread 0.9

АДДИТОР — дополняет входные теги новыми (входные В результате остаются):
  python src/generate.py addit "1girl, long_hair, blue_eyes"
  python src/generate.py addit "1girl, sword" --count 15 --temp 1.2 --spread 0.7

Без входных тегов — генерирует от 3 случайных тегов:
  python src/generate.py abstract
  python src/generate.py addit

Параметры:
  --count   N      количество генерируемых тегов (default: 20)
  --temp    0-2    температура: низкая=близко к запросу, высокая=творчески (default: 1.0)
  --spread  0-1    разброс тем: низкий=близкие темы, высокий=далёкие (default: 0.9)
"""

import sys
import json
import random
import argparse
from pathlib import Path

import torch

BASE_DIR      = Path(__file__).parent.parent
VOCAB_FILE    = BASE_DIR / "data"  / "vocab.json"
CHECKPOINT_F  = BASE_DIR / "model" / "tagger_checkpoint.pth"
BANLIST_FILE   = BASE_DIR / "data"  / "banlist.txt"
WHITELIST_FILE = BASE_DIR / "data"  / "whitelist.txt"
PROFILES_DIR   = BASE_DIR / "data"  / "profiles"

# Переопределяются через --vocab и --checkpoint если переданы
_vocab_override      = None
_checkpoint_override = None



def load_model_and_vocab(vocab_path=None, checkpoint_path=None):
    """Загружает модель и словарь."""
    vocab_f = Path(vocab_path) if vocab_path else VOCAB_FILE
    ckpt_f  = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_F

    if not vocab_f.exists():
        print(f"[ERROR] vocab не найден: {vocab_f}")
        sys.exit(1)
    if not ckpt_f.exists():
        print(f"[ERROR] checkpoint не найден: {ckpt_f}")
        sys.exit(1)

    print(f"  Vocab:      {vocab_f.name}")
    print(f"  Checkpoint: {ckpt_f.name}")

    with open(vocab_f, encoding="utf-8") as f:
        vocab = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_f, map_location=device)

    # Если vocab.json не совпадает с чекпоинтом — берём размер из чекпоинта
    ckpt_vocab_size = ckpt.get("vocab_size", vocab["vocab_size"])
    if ckpt_vocab_size != vocab["vocab_size"]:
        print(f"[INFO] vocab.json ({vocab['vocab_size']:,} токенов) не совпадает с чекпоинтом "
              f"({ckpt_vocab_size:,} токенов).")
        print(f"[INFO] Используется словарь из чекпоинта (первые {ckpt_vocab_size:,} токенов).")
        # Обрезаем vocab до размера чекпоинта
        token2id = {t: i for t, i in vocab["token2id"].items() if i < ckpt_vocab_size}
        id2token = {i: t for i, t in enumerate(list(vocab["token2id"].keys())[:ckpt_vocab_size])}
        vocab = dict(vocab)
        vocab["vocab_size"] = ckpt_vocab_size
    else:
        token2id = vocab["token2id"]
        id2token = {int(k): v for k, v in vocab["id2token"].items()}

    from tagger_model import TagTransformer
    model = TagTransformer(
        vocab_size     = ckpt_vocab_size,
        d_model        = ckpt["d_model"],
        nhead          = ckpt["nhead"],
        num_layers     = ckpt["num_layers"],
        dim_feedforward= ckpt["d_model"] * 4,
        max_seq_len    = ckpt["max_seq_len"],
        pad_id         = vocab["special_tokens"]["PAD"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, vocab, token2id, id2token, device


def parse_tag_file(filepath, section=None):
    """
    Парсит теги из файла — любой формат: запятые, пробелы, новые строки.
    Поддерживает секции [banlist] и [whitelist].
    section=None — читает все теги без учёта секций.
    section='banlist' или 'whitelist' — читает только из нужной секции.
    """
    if not filepath.exists():
        return set()
    with open(filepath, encoding="utf-8") as f:
        raw = f.read()
    tags = set()
    current_section = None

    for line in raw.splitlines():
        # Убираем комментарий
        line = line.split("#")[0].strip()
        if not line:
            continue
        # Проверяем секцию [banlist] / [whitelist]
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].lower()
            continue
        # Если секция задана — читаем только нужную
        if section and current_section != section:
            continue
        # Парсим теги через запятую
        for part in line.split(","):
            tag = part.strip().lower().replace(" ", "_")
            if tag:
                tags.add(tag)
    return tags


def load_profile(profile_name):
    """Загружает banlist и whitelist из профиля в data/profiles/."""
    path = PROFILES_DIR / f"{profile_name}.txt"
    if not path.exists():
        print(f"[WARN] Профиль не найден: {path}")
        return set(), set()
    banlist   = parse_tag_file(path, section="banlist")
    whitelist = parse_tag_file(path, section="whitelist")
    return banlist, whitelist


def text_to_seed_tags(text, token2id):
    """Парсит теги из строки через запятую."""
    raw = [t.strip() for t in text.replace(",", " ").split() if t.strip()]
    found   = [t for t in raw if t in token2id]
    unknown = [t for t in raw if t not in token2id]
    if unknown:
        print(f"[WARN] Не найдены в словаре: {', '.join(unknown)}")
    return found


def get_random_seed(token2id, vocab, n=3):
    """3 случайных тега из популярных."""
    bos = vocab["special_tokens"]["BOS"]
    eos = vocab["special_tokens"]["EOS"]
    pad = vocab["special_tokens"]["PAD"]
    unk = vocab["special_tokens"]["UNK"]
    sep = vocab["special_tokens"]["SEP"]
    special = {bos, eos, pad, unk, sep}

    candidates = [tid for tid in range(len(token2id))
                  if tid not in special]
    # Берём из топ-500 самых частых тегов
    top_candidates = candidates[:500]
    return random.sample(top_candidates, min(n, len(top_candidates)))


def print_tags(tags, title, mode, temp, spread, count):
    """Красивый вывод результата."""
    print(f"\n{'='*60}")
    print(f"  Режим:       {mode}")
    print(f"  Температура: {temp}  |  Разброс: {spread}  |  Кол-во: {count}")
    print(f"{'='*60}")
    if not tags:
        print("  Нет результатов.")
        return
    print(f"  Всего тегов: {len(tags)}")
    print(f"{'='*60}")
    print(f"\n>>> COPY TAGS <<<")
    print(", ".join(tags))
    print(f">>> END <<<\n")


def generate_tags(model, vocab, token2id, id2token, device,
                  seed_ids, count, temperature, spread, mode,
                  input_tags=None, banlist=None, whitelist=None):
    """Основная функция генерации."""
    bos_id = vocab["special_tokens"]["BOS"]
    sep_id = vocab["special_tokens"]["SEP"]
    eos_id = vocab["special_tokens"]["EOS"]

    if mode == "abstract":
        input_ids = [bos_id] + seed_ids
    else:
        input_ids = [bos_id] + seed_ids + [sep_id]

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Запрещаем повтор входных тегов в Аддитор режиме
    forbidden = set(seed_ids) if mode == "addit" else set()

    # Добавляем banlist в forbidden
    if banlist:
        for tag in banlist:
            tid = token2id.get(tag)
            if tid is not None:
                forbidden.add(tid)

    # Whitelist — разрешаем только эти теги (запрещаем все остальные)
    whitelist_ids = None
    if whitelist:
        whitelist_ids = {token2id[t] for t in whitelist if t in token2id}

    new_ids = model.generate(
        input_ids     = input_tensor,
        vocab         = vocab,
        max_new_tags  = count,
        temperature   = temperature,
        top_p         = spread,
        forbidden_ids = forbidden,
        whitelist_ids = whitelist_ids,
    )

    # Декодируем
    result_tags = []
    if mode == "addit" and input_tags:
        result_tags = list(input_tags)  # Аддитор: входные теги + новые
    # Абстракт: только новые теги, входные не включаем

    for tid in new_ids:
        tag = id2token.get(tid, None)
        if tag and not tag.startswith("<"):
            result_tags.append(tag)

    return result_tags


def main():
    parser = argparse.ArgumentParser(
        description="Генерация тегов в стиле Danbooru",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mode", nargs="?", choices=["abstract", "addit"],
                        default="abstract",
                        help="abstract = по описанию | addit = дополнить теги")
    parser.add_argument("input", nargs="?", default="",
                        help="Текст (abstract) или теги через запятую (addit)")
    parser.add_argument("--count",  type=int,   default=20,
                        help="Количество генерируемых тегов (default: 20)")
    parser.add_argument("--temp",   type=float, default=1.0,
                        help="Температура 0.1-2.0: низкая=предсказуемо (default: 1.0)")
    parser.add_argument("--spread", type=float, default=0.9,
                        help="Разброс тем 0.1-1.0: низкий=близко, высокий=далёко (default: 0.9)")
    parser.add_argument("--ban",     action="store_true",
                        help="Включить banlist из data/banlist.txt")
    parser.add_argument("--white",   action="store_true",
                        help="Включить whitelist из data/whitelist.txt")
    parser.add_argument("--profile", nargs="+", default=[],
                        help="Профили из data/profiles/ (можно несколько: --profile ud_age ud_violence)")
    parser.add_argument("--batch",   type=int, default=1,
                        help="Количество генераций подряд (default: 1)")
    parser.add_argument("--vocab",      type=str, default=None,
                        help="Путь к vocab.json (default: data/vocab.json)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Путь к checkpoint .pth (default: model/tagger_checkpoint.pth)")
    parser.add_argument("--sort", action="store_true",
                        help="Сортировать теги по алфавиту")
    args = parser.parse_args()

    print("Загружаю модель...")
    model, vocab, token2id, id2token, device = load_model_and_vocab(
        vocab_path      = args.vocab,
        checkpoint_path = args.checkpoint,
    )
    print(f"Устройство: {device}")

    # Загружаем banlist и whitelist
    banlist   = parse_tag_file(BANLIST_FILE)   if args.ban   else set()
    whitelist = parse_tag_file(WHITELIST_FILE)  if args.white else set()

    # Загружаем профили (можно несколько)
    for profile_name in args.profile:
        b, w = load_profile(profile_name)
        banlist   |= b
        whitelist |= w
        print(f"Профиль '{profile_name}': +{len(b)} ban, +{len(w)} white")

    if banlist:
        print(f"Banlist итого:   {len(banlist)} тегов")
    if whitelist:
        print(f"Whitelist итого: {len(whitelist)} тегов")

    mode  = args.mode
    text  = args.input.strip()

    # Получаем seed теги
    if not text:
        print("Входные данные не указаны — генерирую 3 случайных seed тега...")
        seed_ids   = get_random_seed(token2id, vocab, n=3)
        input_tags = [id2token[i] for i in seed_ids]
        print(f"Seed теги: {', '.join(input_tags)}")
    elif mode == "abstract":
        input_tags = text_to_seed_tags(text, token2id)
        if not input_tags:
            print(f"[WARN] Не нашёл подходящих тегов для '{text}', использую случайные")
            seed_ids   = get_random_seed(token2id, vocab, n=3)
            input_tags = [id2token[i] for i in seed_ids]
        else:
            seed_ids = [token2id[t] for t in input_tags]
        print(f"Seed теги (из описания): {', '.join(input_tags)}")
    else:
        # Аддитор — входные теги
        raw_tags   = [t.strip() for t in text.split(",") if t.strip()]
        input_tags = [t for t in raw_tags if t in token2id]
        unknown    = [t for t in raw_tags if t not in token2id]
        if unknown:
            print(f"[WARN] Теги не найдены в словаре: {', '.join(unknown)}")
        if not input_tags:
            print("[WARN] Нет валидных тегов, использую случайные")
            seed_ids   = get_random_seed(token2id, vocab, n=3)
            input_tags = [id2token[i] for i in seed_ids]
        else:
            seed_ids = [token2id[t] for t in input_tags]
        print(f"Входные теги: {', '.join(input_tags)}")

    # Генерация (одиночная или батч)
    all_tags_pool = []  # все теги со всех генераций для финального списка

    for i in range(args.batch):
        if args.batch > 1:
            print(f"\n[{i+1}/{args.batch}] Generating...")

        result = generate_tags(
            model, vocab, token2id, id2token, device,
            seed_ids    = seed_ids,
            count       = args.count,
            temperature = args.temp,
            spread      = args.spread,
            mode        = mode,
            input_tags  = input_tags if mode == "addit" else None,
            banlist     = banlist,
            whitelist   = whitelist if whitelist else None,
        )

        if args.sort:
            result = sorted(result)

        if args.batch == 1:
            print_tags(result, text, mode.upper(), args.temp, args.spread, args.count)
        else:
            # В батч-режиме тихо собираем каждую генерацию отдельно
            all_tags_pool.append(result)

    # Финальный вывод после батча
    if args.batch > 1:
        print(f"\n{'='*60}")
        print(f"  Режим: {mode.upper()}  |  Генераций: {args.batch}")
        print(f"{'='*60}")
        print(f"\n>>> COPY TAGS <<<\n")
        for i, tags in enumerate(all_tags_pool, 1):
            print(f"[{i}] " + ", ".join(tags))
            print()
        print(f">>> END <<<\n")


if __name__ == "__main__":
    main()
