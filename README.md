## Micro-Tokenizers

Inspired by GPT-style tokenization, **Micro-Tokenizers** is an educational, from-scratch implementation of **byte-level BPE** and a **regex-splitting tokenizer** in pure Python. The goal is to make tokenization transparent: how raw UTF‑8 bytes become subword tokens through merges, and how GPT-like regex chunking + special tokens fit into the picture.

## 🏗️ Architecture Overview

### Byte Pair Encoding (BPE) — Byte-level merges
- Starts from a **256-byte vocabulary** (0–255)
- Learns new tokens by repeatedly merging the **most frequent adjacent byte-pair**
- Great for understanding how a subword vocabulary is built from data

### Regex Tokenizer (GPT-like)
- Uses GPT-style **regex splitting** (GPT‑2 / GPT‑4 patterns)
- Trains BPE merges over the resulting text chunks
- Supports **special tokens** (e.g., `<|endoftext|>`, FIM tokens)

### Save / Load (Reproducible tokenization)
- Saves a compact `.model` (pattern + merges + special tokens)
- Saves a human-readable `.vocab` for inspection/debugging

## Learning Resources

### Scripts
- **`train.py`**: trains both tokenizers on the included corpus (`taylorswift.txt`) and saves models to `models/`

### Tests (as documentation)
- **`test.py`**: pytest suite that checks:
  - encode/decode identity
  - a Wikipedia BPE example
  - save/load roundtrip
  - (includes `tiktoken` as a reference dependency)

## Core Components
- **Pair statistics + merge**: count adjacent token pairs and replace them with a new token id
- **Tokenizer base class**: shared training/encode/decode interface + vocab construction
- **Regex splitting**: GPT‑2 / GPT‑4 split patterns for realistic tokenization
- **Special tokens**: register reserved ids and allow/deny them during encoding
- **Persistence**: export/import a trained tokenizer via `.model`

## 🛠️ Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```
### Train tokenizers (and save models)
```bash
python train.py
```
This will create:
- models/basic.model + models/basic.vocab
- /regex.model + models/regex.vocab
### Run tests
```bash
pip install pytest
pytest -v
```
### 🤝 Contributing
- Contributions are welcome — this is an educational project, so please focus on:
- Code clarity and comments
- Tests and small reproducible examples
- Bug fixes and correctness improvements
- Better documentation / diagrams explaining BPE + regex splitting

### 📄 License
This project is open source and available under the MIT License.
> Note: This is an educational implementation. For production use, consider established libraries like Hugging Face Tokenizers / Transformers or OpenAI’s tiktoken.

