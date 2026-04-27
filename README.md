# Danbooru Tagger — ForgeNeo Extension

Расширение добавляет в txt2img/img2img аккордеон **"🏷️ Danbooru Tagger"**
(аналогично ADetailer). При включении генерирует теги через твою DanbooruAI-модель
и вставляет их в промпт перед диффузией.

---

## Быстрый старт

### 1. Установи расширение

Скопируй папку `danbooru-tagger` в директорию расширений ForgeNeo:

```
ForgeNeo/
  extensions/
    danbooru-tagger/   ← сюда
```

---

### 2. Файлы модели и словаря

Уже лежат внутри расширения (скопированы автоматически):

```
danbooru-tagger/
  data/
    vocab_clean.json      ✓
  model/
    tagger_clean.pth      ✓
```

---

### 3. Python-файлы модели

Тоже уже скопированы:

```
danbooru-tagger/
  tagger_lib/
    generate.py           ✓  (из DanbooruAI/src/)
    tagger_model.py       ✓  (из DanbooruAI/src/)
    inference.py          ✓  (обёртка расширения)
```

---

### 4. Safety-профили (необязательно)

Профили — это `.txt`-файлы с секциями `[banlist]` и `[whitelist]`.

Скопируй из `DanbooruAI/data/profiles/` в:

```
danbooru-tagger/
  data/
    profiles/
      ud_age.txt
      ud_animals.txt
      ud_violence.txt
```

Формат файла:

```
[banlist]
tag_one, tag_two
another_tag

[whitelist]
allowed_tag
```

Если профилей нет — фильтрация просто не применяется.

---

## Итоговая структура

```
danbooru-tagger/
├── scripts/
│   └── danbooru_tagger.py     ← главный скрипт Forge (UI + хук)
├── tagger_lib/
│   ├── __init__.py
│   ├── inference.py           ← singleton-обёртка
│   ├── generate.py            ← из DanbooruAI/src/
│   └── tagger_model.py        ← из DanbooruAI/src/
├── data/
│   ├── vocab_clean.json
│   └── profiles/              ← необязательно
├── model/
│   └── tagger_clean.pth
├── install.py
└── README.md
```

---

## Параметры UI

| Параметр | Описание |
|----------|----------|
| **Enable** | Включить расширение |
| **Mode** | `abstract` — теги по затравкам (без затравок в выводе) / `addit` — расширить промпт |
| **Вставить в промпт** | `prepend` — перед промптом / `append` — после / `replace` — заменить |
| **Seed tags** | Входные теги (через запятую). Пусто = взять из текущего промпта |
| **Кол-во тегов** | Количество генерируемых тегов |
| **Temperature** | 0.1 = предсказуемо / 2.0 = творчески |
| **Spread** | Nucleus p: 0.1 = близкие темы / 1.0 = далёкие |
| **Сортировать теги** | Алфавитная сортировка |
| **Safety profiles** | ud_age / ud_animals / ud_violence |
| **🔍 Preview tags** | Генерация без запуска диффузии |

---

## Отладка

Все сообщения начинаются с `[DanbooruTagger]` — смотри консоль ForgeNeo.

- Модель не загружается → проверь пути в консоли
- `tagger_model.py` не найден → убедись что он в `tagger_lib/`
- Профиль не найден → расширение продолжит работу без него
