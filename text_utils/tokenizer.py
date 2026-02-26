"""
CLIP BPE Tokenizer.
Based on the original OpenAI CLIP tokenizer implementation.
Downloads BPE merges on first use and caches locally.
"""

import gzip
import html
import os
import urllib.request
from functools import lru_cache

import ftfy
import regex as re
import torch


_BPE_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
_BPE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "clip")


@lru_cache()
def _default_bpe():
    os.makedirs(_BPE_CACHE_DIR, exist_ok=True)
    bpe_path = os.path.join(_BPE_CACHE_DIR, "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(bpe_path):
        print(f"Downloading CLIP BPE vocab to {bpe_path} ...")
        urllib.request.urlretrieve(_BPE_URL, bpe_path)
    return bpe_path


@lru_cache()
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def _get_pairs(word):
    pairs = set()
    prev = word[0]
    for char in word[1:]:
        pairs.add((prev, char))
        prev = char
    return pairs


def _basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SimpleTokenizer:
    def __init__(self, bpe_path=None):
        bpe_path = bpe_path or _default_bpe()
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1: 49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]

        vocab = list(_bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"""
            r"""|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = _whitespace_clean(_basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join(self.decoder[t] for t in tokens)
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text


_tokenizer = SimpleTokenizer()


def tokenize(texts, context_length=77, truncate=True):
    """Tokenize a string or list of strings and return a padded tensor."""
    if isinstance(texts, str):
        texts = [texts]

    sot = _tokenizer.encoder["<|startoftext|>"]
    eot = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot] + _tokenizer.encode(text) + [eot] for text in texts]

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot
            else:
                raise RuntimeError(
                    f"Input is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = torch.tensor(tokens)
    return result
