
import json
import re

import regex as re2  # optional dependency

GPT2_PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def _pretokenize_gpt2(text):
    # same behavior as in your trainer loop
    # returns list of bytes segments
    toks = re2.findall(GPT2_PAT, text)
    return [t.encode("utf-8") for t in toks if t]

def _remove_and_split_specials(text: str, special_tokens: list[str]) -> list[str]:
   """Remove and split on special tokens so merges never cross them."""
   if not special_tokens:
       return [text]
   escaped = [re.escape(tok) for tok in special_tokens]
   pattern = "|".join(escaped)
   # Split while preserving text segments between tokens
   parts = re.split(pattern, text)
   return [p for p in parts if p.strip() != ""]

def _token_str_to_bytes(s: str) -> bytes:
    # Map the common BPE marker "Ġ" (U+0120) to a literal space.
    # As long as we do it for both vocab and merges, internally they are matched.
    # So our input string like " cat" can be properly translated.

    return s.replace("Ġ", " ").encode("utf-8")

class Tokenizer:
    def __init__(self, vocab: dict, merges: list, special_tokens: list | None = None):
        # vocab: dict[int, bytes]
        # merges: list[tuple[bytes, bytes]]
        # special_tokens: list[str] | None = None

        # self.id_to_bytes: dict[int, bytes]
        # self.bytes_to_id: dict[bytes, int]
        # self.merges: list[tuple[bytes, bytes]]
        self.id_to_bytes = dict(vocab)
        self.bytes_to_id = {b: i for i, b in self.id_to_bytes.items()}
        self.merges = merges

        # handling special tokens
        # if input text has <|endoftext|> we should encode/encode it as atomic token
        self.special_tokens = special_tokens or []

        # Ensure all specials are in vocab
        # E.g. we have double end-of-text as special token


        for s in self.special_tokens:
            sb = s.encode("utf-8")
            if sb not in self.bytes_to_id:
                new_id = max(self.id_to_bytes.keys()) + 1
                self.id_to_bytes[new_id] = sb
                self.bytes_to_id[sb] = new_id

        # Compile regex for splitting on specials
        if self.special_tokens:
            escaped = [re.escape(s) for s in self.special_tokens]
            # longest first so the double marker wins over the single
            escaped.sort(key=len, reverse=True)
            self._special_split_re = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_split_re = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list | None = None):

        # vocab.json format: { "token": id, ... }
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            tok2id = json.load(f)
        vocab = {}
        for tok_str, tid in tok2id.items():
            b = _token_str_to_bytes(tok_str)
            vocab[tid] = b

        # merges.txt format: one pair per line: "A B" (tokens may use "Ġ" for space)
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip() # Ġ t
                parts = s.split()
                if len(parts) != 2:
                    raise ValueError("bad merges line: {}".format(s))
                a_str, b_str = parts
                a = _token_str_to_bytes(a_str)
                b = _token_str_to_bytes(b_str)
                merges.append((a, b))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:

        # text: the cat ate
        # pretokenized: [b'the', b' cat', b' ate']
        ids = []

        # before doing gpt2 pretokenization, use special token re to split normal words with special tokens
        # "Hi<|endoftext|>there" -> ["Hi", "<|endoftext|>", "there"]

        # 'Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>'
        # =>
        # [('Héllò hôw ', False), ('<|endoftext|>', True), ('', False), ('<|endoftext|>', True), (' are ü? 🙃', False),
        #  ('<|endoftext|>', True), ('', False)]
        split_specials = self._split_specials(text)
        for segment, is_special in split_specials:
            if is_special:
                sb = segment.encode("utf-8")
                ids.append(self.bytes_to_id[sb])
                continue
            pretokenized = _pretokenize_gpt2(segment)
            for pretok in pretokenized:
                merged_chunks = self._apply_merges(pretok)
                ids.extend(self._map_chunks_to_ids(merged_chunks))
        return ids

    def encode_iterable(self, iterable):
        """
        Lazily yield token IDs for each input string in `iterable`.
        Each element of `iterable` is pretokenized and encoded independently,
        so tokens never cross chunk boundaries (streaming-friendly).
        """
        for text in iterable:
            for segment, is_special in self._split_specials(text):
                if is_special:
                    sb = segment.encode("utf-8")
                    yield self.bytes_to_id[sb]
                else:
                    pretokenized = _pretokenize_gpt2(segment)
                    for pretok in pretokenized:
                        merged = self._apply_merges(pretok)
                        for tid in self._map_chunks_to_ids(merged):
                            yield tid

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for tid in ids:
            b = self.id_to_bytes[tid] # TODO: invalid input
            buf.extend(b)
        return bytes(buf).decode("utf-8", errors="replace")

    def _apply_merges(self, pretok) -> list[bytes]:
        # pretok: b'the'
        # output: [b'th', b'e']
        seq = [pretok[i:i + 1] for i in range(len(pretok))] # [b't', b't', b'e']
        if not seq:
            return []

        changed = True
        while changed:
            changed = False
            # walk merge rules in creation order
            # Once we find the first merge break, and retry merge again
            for a, b in self.merges:
                i = 0
                while i < len(seq) - 1:
                    if seq[i] == a and seq[i + 1] == b:
                        # merge adjacent pair
                        seq[i:i + 2] = [a + b]
                        changed = True
                        # restart from first merge rule
                        i = None
                        break
                    i += 1
                if i is None:
                    break
        return seq

    def _split_specials(self, text):
        """Return [(segment, is_special: bool), ...]."""
        # no special tokens?
        if not self._special_split_re:
            return [(text, False)]
        parts = self._special_split_re.split(text)
        out = []
        special_set = set(self.special_tokens)
        for p in parts:
            out.append((p, p in special_set))
        return out
    def _map_chunks_to_ids(self, chunks) -> list[int]:
        out = []
        for tok in chunks:
            # Assumption: all tok bytes exist in vocab?
            # See if we get key error here.
            tid = self.bytes_to_id[tok]
            out.append(tid)
        return out