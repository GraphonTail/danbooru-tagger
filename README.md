# Danbooru Tagger

An AI-powered tag generator for **Stable Diffusion Forge / ForgeNeo**, trained on **1 million (future 11.8 m) Danbooru posts**.

Generates contextual booru-style tags from your prompt before each diffusion run, enriching and diversifying the final image.

---

## How it works

```
Your prompt  →  Tagger  →  Extended prompt  →  Stable Diffusion  →  Image
```

The tagger samples tags conditioned on your existing prompt and merges them back in before diffusion runs. Use **Preview tags** to inspect the output without generating an image — results can be copied to clipboard or injected directly into the prompt field.

---

## Installation

**From URL** *(recommended)*

1. Open **Forge → Extensions → Install from URL**
2. Paste `https://github.com/GraphonTail/danbooru-tagger`
3. Click **Install**, then **Apply and restart UI**

**Manual**

```bash
cd /path/to/forge/extensions
git clone https://github.com/GraphonTail/danbooru-tagger
```

---

## Model files

The weights are too large for GitHub. Download them from the **[Releases page](https://github.com/GraphonTail/danbooru-tagger/releases)** and place them here:

```
danbooru-tagger/
├── model/
│   └── tagger_clean.pth        ← download from Releases
└── data/
    └── vocab_clean.json        ← download from Releases
```

On the next Forge launch, `install.py` will print a warning with the exact expected paths if either file is missing.

---

## UI controls

| Control | Description |
|---------|-------------|
| **Enable** | Header toggle — enables or disables the tagger for this generation |
| **Model** | Select any `.pth` file from the `model/` folder |
| **Insert mode** | How tags are merged into your prompt: `prepend` (before) or `append` (after) |
| **Tag count** | Number of tags to generate (1 – 500) |
| **Temperature** | Creativity / randomness (0.1 – 50). Recommended: 0.75 – 1.25 |
| **Spread** | Topic diversity (0.1 – 50). Recommended: 1.0 – 2.0 |
| **Safety profile** | Optional tag filter — select a `.txt` profile from `data/profiles/` |
| **Preview tags** | Generate and display tags without running diffusion |

The **Preview** panel includes two buttons:
- **Copy** — copies the generated tags to clipboard
- **Inject** — inserts the tags directly into the active prompt field

---

## Safety profiles

Create `.txt` files in `data/profiles/` to block or explicitly allow specific tags:

```
data/profiles/
└── my_profile.txt
```

```ini
[banlist]
gore, violence, nsfw_tag

[whitelist]
safe_tag
```

Use the **⟳ Refresh** button to rescan the folder without restarting Forge.

---

## Multiple models

Drop additional `.pth` files into `model/` and their matching vocabulary files into `data/`. The **Model** dropdown lists all available weights — no restart required.

**Planned variants:** General · Safe · Explicit · Furry · Chaos

---

## Requirements

- Stable Diffusion Forge or ForgeNeo
- Python 3.10+
- PyTorch (bundled with Forge — no separate install needed)

---

## License

[MIT](LICENSE)
