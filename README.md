# Micro-Tokenizers
Micro-Tokenizers is a pure Python implementation of how modern LLM tokenization works: it learns byte-level BPE merges (starting from the 256 UTF‑8 bytes) and also includes a GPT-style regex tokenizer with special-token support, plus simple save/load so you can train, inspect, and reuse vocabularies.
