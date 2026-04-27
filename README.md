# Danbooru Tagger — ForgeNeo Extension

This extension adds a **"🏷️ Danbooru Tagger"** accordion to txt2img/img2img  
(similar to ADetailer). When enabled, it generates tags using your DanbooruAI model  
and inserts them into the prompt before diffusion.

---

## Quick Start

### 1. Install the extension

Copy the `danbooru-tagger` folder into the ForgeNeo extensions directory:

```
ForgeNeo/
  extensions/
    danbooru-tagger/   ← place it here
```

---

### 2. Model and vocabulary files

Already included inside the extension (copied automatically):

```
danbooru-tagger/
  data/
    vocab_clean.json      ✓
  model/
    tagger_clean.pth      ✓
```

---

### 3. Model Python files

Also already included:

```
danbooru-tagger/
  tagger_lib/
    generate.py
    tagger_model.py
    inference.py
```

---

### 4. Safety profiles (optional)

Profiles are `.txt` files with `[banlist]` and `[whitelist]` sections.

Copy them from `DanbooruAI/data/profiles/` into:

```
danbooru-tagger/
  data/
    profiles/
      ud_age.txt
      ud_animals.txt
      ud_violence.txt
```

File format:

```
[banlist]
tag_one, tag_two
another_tag

[whitelist]
allowed_tag
```

If no profiles are provided, filtering will simply not be applied.

---

## Final structure

```
danbooru-tagger/
├── scripts/
│   └── danbooru_tagger.py     ← main Forge script (UI + hook)
├── tagger_lib/
│   ├── __init__.py
│   ├── inference.py           
│   ├── generate.py            
│   └── tagger_model.py        
├── data/
│   ├── vocab_clean.json
│   └── profiles/              ← optional
├── model/
│   └── tagger_clean.pth
├── install.py
└── README.md
```

---

## UI Parameters

| Parameter              | Description                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| **Enable**             | Enable the extension                                                                         |
| **Mode**               | `abstract` — generate tags from seeds (not included in output) / `addit` — extend the prompt |
| **Insert into prompt** | `prepend` — before prompt / `append` — after / `replace` — replace                           |
| **Seed tags**          | Input tags (comma-separated). Empty = use current prompt                                     |
| **Tag count**          | Number of generated tags                                                                     |
| **Temperature**        | 0.1 = predictable / 2.0 = creative                                                           |
| **Spread**             | Nucleus p: 0.1 = close topics / 1.0 = distant                                                |
| **Sort tags**          | Alphabetical sorting                                                                         |
| **Safety profiles**    | ud_age / ud_animals / ud_violence                                                             |
| **🔍 Preview tags**    | Generate without running diffusion                                                           |

---

## Debugging

All messages start with `[DanbooruTagger]` — check the ForgeNeo console.

- Model not loading → check paths in console
- `tagger_model.py` not found → make sure it exists in `tagger_lib/`
- Profile not found → extension will continue without it