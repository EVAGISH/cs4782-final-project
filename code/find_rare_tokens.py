"""Find rare single-token identifiers in the SD-v1.5 CLIP tokenizer.

DreamBooth needs an identifier (the "[V]" placeholder in the paper, e.g. "sks")
that satisfies three properties:

  1. Encodes as exactly 1 token in the prompt.
     Multi-token identifiers like "[V]" tokenize to 3+ pieces, fragmenting the
     subject binding across unrelated tokens — a known DreamBooth failure mode.

  2. Has a high vocab ID.
     CLIP's BPE vocab is built greedily from most-frequent merges first; tokens
     near the end of the vocab (high ID) are the rarest and have the weakest
     prior associations the model has to overwrite.

  3. Is short (1-4 chars) and pronounceable.
     Easier to use in prompts; harder for the model to confuse with real words.

Usage:
    python code/find_rare_tokens.py
    python code/find_rare_tokens.py --top_k 30 --max_len 3
    python code/find_rare_tokens.py --check sks zwx qx
"""

import argparse
import re

from transformers import CLIPTokenizer


COMMON_ENGLISH = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "man", "new", "now", "old", "see", "two", "way", "who",
    "boy", "did", "its", "let", "put", "say", "she", "too", "use", "yet",
    "yes", "ago", "act", "add", "age", "air", "all", "ask", "bad", "bag",
    "big", "box", "bus", "buy", "car", "cat", "cup", "cut", "dad", "die",
    "dog", "eat", "end", "eye", "far", "fat", "fly", "fun", "god", "got",
    "gun", "guy", "hat", "hit", "hot", "ice", "job", "key", "kid", "lay",
    "leg", "let", "lie", "lot", "low", "map", "may", "mom", "off", "oil",
    "own", "pay", "pen", "pop", "red", "run", "sad", "sea", "set", "sex",
    "sit", "ski", "sky", "son", "sun", "tax", "tea", "ten", "tip", "top",
    "try", "war", "win", "yet", "yes", "are", "art", "bed", "bee", "bit",
    "bow", "cow", "ear", "egg", "ego", "elf", "fee", "fig", "fix", "fox",
    "gas", "gel", "gem", "gum", "hen", "hop", "hub", "hug", "hum", "ink",
    "jam", "jar", "jaw", "jet", "joy", "lid", "log", "mad", "mat", "mid",
    "mix", "mug", "nap", "net", "nut", "oar", "odd", "ore", "owl", "pad",
    "pan", "pat", "paw", "pet", "pig", "pin", "pit", "pot", "pup", "rag",
    "rat", "raw", "rib", "rim", "rod", "rot", "rub", "rug", "sip", "sir",
    "sob", "sow", "spa", "tag", "tan", "tap", "tar", "tax", "tow", "toy",
    "tub", "wig", "yam", "zip", "zoo",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--max_len", type=int, default=4,
                        help="Max character length of identifier (default 4)")
    parser.add_argument("--min_len", type=int, default=2,
                        help="Min character length (default 2)")
    parser.add_argument("--min_id", type=int, default=30000,
                        help="Only consider tokens with vocab ID >= this (default 30000)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="How many candidates to print (default 40)")
    parser.add_argument("--check", type=str, nargs="*", default=None,
                        help="Specific tokens to inspect: prints how each is tokenized")
    return parser.parse_args()


def strip_eow(token_str):
    """CLIP tokenizer marks end-of-word with </w>. Return (bare_string, is_eow)."""
    if token_str.endswith("</w>"):
        return token_str[: -len("</w>")], True
    return token_str, False


def tokenize_phrase(tok, identifier, class_noun="dog"):
    """Encode 'a photo of {identifier} {class_noun}' and return per-piece decoding."""
    phrase = f"a photo of {identifier} {class_noun}"
    ids = tok.encode(phrase, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    return phrase, ids, pieces


def encodes_as_single(tok, identifier):
    """Check whether the identifier alone encodes to exactly 1 token."""
    return len(tok.encode(identifier, add_special_tokens=False)) == 1


def encodes_as_single_in_phrase(tok, identifier, class_noun="dog"):
    """Check whether, inside a real prompt, the identifier still occupies a
    single token slot. This is what DreamBooth cares about."""
    base = tok.encode(f"a photo of {class_noun}", add_special_tokens=False)
    full = tok.encode(f"a photo of {identifier} {class_noun}",
                      add_special_tokens=False)
    return len(full) - len(base) == 1


def main():
    args = parse_args()
    tok = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    vocab = tok.get_vocab()
    print(f"Loaded tokenizer with vocab size {len(vocab)} from {args.model}")

    if args.check:
        print(f"\n=== Inspecting specific identifiers ===\n")
        for tid in args.check:
            phrase, ids, pieces = tokenize_phrase(tok, tid)
            single_alone = encodes_as_single(tok, tid)
            single_in_phrase = encodes_as_single_in_phrase(tok, tid)
            print(f"  '{tid}':")
            print(f"    standalone tokens : {tok.encode(tid, add_special_tokens=False)}  "
                  f"({'OK 1 token' if single_alone else 'FRAGMENTED'})")
            print(f"    in 'a photo of {tid} dog' : {ids}")
            print(f"    pieces             : {pieces}")
            print(f"    single in phrase?  : {'YES' if single_in_phrase else 'NO'}")
            print()
        return

    print(f"\n=== Top {args.top_k} candidate rare identifiers ===")
    print(f"Filters: vocab_id >= {args.min_id}, length {args.min_len}-{args.max_len}, "
          f"alphabetic, not common English\n")

    candidates = []
    for token_str, token_id in vocab.items():
        if token_id < args.min_id:
            continue
        bare, is_eow = strip_eow(token_str)
        if not is_eow:
            continue
        if not (args.min_len <= len(bare) <= args.max_len):
            continue
        if not re.fullmatch(r"[a-z]+", bare):
            continue
        if bare in COMMON_ENGLISH:
            continue
        if not encodes_as_single(tok, bare):
            continue
        if not encodes_as_single_in_phrase(tok, bare):
            continue
        candidates.append((token_id, bare))

    candidates.sort(reverse=True)

    print(f"{'rank':>4}  {'vocab_id':>8}  {'token':<6}  example tokenization")
    print("-" * 80)
    for rank, (tid, bare) in enumerate(candidates[: args.top_k], 1):
        _, _, pieces = tokenize_phrase(tok, bare)
        pieces_repr = " | ".join(repr(p) for p in pieces)
        print(f"{rank:>4}  {tid:>8}  {bare:<6}  {pieces_repr}")

    if candidates:
        print(f"\nTotal qualifying candidates: {len(candidates)}")
        print(f"Rarest:    '{candidates[0][1]}'  (vocab_id {candidates[0][0]})")
        print(f"Reference: 'sks' is the paper's canonical choice; check it with "
              f"--check sks")


if __name__ == "__main__":
    main()
