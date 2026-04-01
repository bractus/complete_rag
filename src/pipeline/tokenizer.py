from transformers import AutoTokenizer

from src.pipeline.steps import TokenizationStep

_tokenizer = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def tokenize(text: str) -> TokenizationStep:
    tok = _get_tokenizer()
    tokens = tok.tokenize(text)
    encoding = tok.encode(text)
    vocab_size = tok.vocab_size
    return TokenizationStep(tokens=tokens, token_ids=list(encoding), vocabulary_size=vocab_size)
