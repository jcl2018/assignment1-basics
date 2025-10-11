

from __future__ import annotations
from typing import Dict, Tuple, List, Iterable
import time


import io
import os
import re
import regex as re2  # optional dependency


GPT2_PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


Word = Tuple[bytes, ...]
Pair = Tuple[bytes, bytes]


### Helper function from
import concurrent.futures as _cf


def find_chunk_boundaries(
   file: io.BufferedReader,
   desired_num_chunks: int,
   split_special_token: bytes,
) -> List[int]:
   assert isinstance(split_special_token, bytes), "split_special_token must be bytes"


   file.seek(0, os.SEEK_END)
   file_size = file.tell()
   file.seek(0)


   if desired_num_chunks <= 0 or file_size == 0:
       return [0, file_size]


   chunk_size = file_size // desired_num_chunks
   chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
   chunk_boundaries[-1] = file_size


   mini_chunk_size = 4096


   for bi in range(1, len(chunk_boundaries) - 1):
       initial_position = chunk_boundaries[bi]
       file.seek(initial_position)
       while True:
           mini_chunk = file.read(mini_chunk_size)
           if mini_chunk == b"":
               chunk_boundaries[bi] = file_size
               break
           found_at = mini_chunk.find(split_special_token)
           if found_at != -1:
               chunk_boundaries[bi] = initial_position + found_at
               break
           initial_position += mini_chunk_size


   return sorted(set(chunk_boundaries))


#### Pretokenization helpers




def _remove_and_split_specials(text: str, special_tokens: List[str]) -> List[str]:
   """Remove and split on special tokens so merges never cross them."""
   if not special_tokens:
       return [text]
   escaped = [re.escape(tok) for tok in special_tokens]
   pattern = "|".join(escaped)
   # Split while preserving text segments between tokens
   parts = re.split(pattern, text)
   return [p for p in parts if p.strip() != ""]


def _count_word_freq_bytes(data: bytes, special_tokens: List[str]) -> Dict[Word, int]:


   text = data.decode("utf-8", errors="ignore")
   segments = _remove_and_split_specials(text, special_tokens)
   freqs: Dict[Word, int] = {}


   for seg in segments:
       toks = re2.findall(GPT2_PAT, seg)
       for t in toks:
           b = t.encode("utf-8")
           if not b:
               continue
           tup = tuple(b[i:i+1] for i in range(len(b)))
           freqs[tup] = freqs.get(tup, 0) + 1
   return freqs


def _count_word_freq_from_slice(args: Tuple[str, int, int,  List[str]]) -> Dict[Word, int]:
   path, start, end, specials = args
   with io.open(path, "rb") as f:
       f.seek(start)
       data = f.read(end - start)
   return _count_word_freq_bytes(data, specials)




def _merge_freq_dicts(dicts: List[Dict[Word, int]]) -> Dict[Word, int]:
   out: Dict[Word, int] = {}
   for d in dicts:
       for k, v in d.items():
           out[k] = out.get(k, 0) + v
   return out
####


def _read_and_pretokenize_chunked(
   path: str,
   desired_num_chunks: int,
   split_special_token: bytes,
   workers: int | None = None,
) -> Dict[Word, int]:
   """
   Build word frequencies by splitting the file into chunks that align on
   split_special_token and counting each chunk in parallel.
   """
   if workers is None:
       workers = max(1, (os.cpu_count() or 1))


   # List of separation positions in files
   with io.open(path, "rb") as f:
       boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token)
   print(boundaries)


   tasks: List[Tuple[str, int, int, List[str]]] = [
       (path, s, e, [split_special_token.decode('utf-8')]) for s, e in zip(boundaries[:-1], boundaries[1:])
   ]


   if workers == 1 or len(tasks) == 1:
       parts = [_count_word_freq_from_slice(t) for t in tasks]
   else:
       with _cf.ProcessPoolExecutor(max_workers=workers) as ex:
           parts = list(ex.map(_count_word_freq_from_slice, tasks))


   return _merge_freq_dicts(parts)




#########################




def _read_and_pretokenize(path: str) -> Dict[Word, int]:
   """
   Read file, split on whitespace, return frequency dict mapping
   Word (tuple of bytes tokens) -> count.
   Each byte token starts as a single-byte bytes object.


   string: hello hello world
   output: {(b'h',b'e',b'l',b'l',b'o'): 2, (b'w',b'o',b'r',b'l',b'd'): 1}


   """
   freqs: Dict[Word, int] = {}


   with io.open(path, "rb") as f:
       data = f.read()
   # Split on ASCII whitespace using regex over bytes.
   # Python's re can handle bytes; ensure pattern is bytes.
   # We compiled pattern on str; convert by decoding then re-encoding would be messy.
   # Instead, split manually for bytes: treat any b"\x00-\x20" as whitespace? Simpler: decode to latin1.
   # But we want a strict whitespace split. Use bytes.split() which splits on any ASCII whitespace by default? No.
   # bytes.split() with None splits on runs of ASCII whitespace (like str.split(None)). That is what we need.
   words = data.split()


   for w in words:
       # Represent word as tuple of byte tokens (initially single bytes)
       tup = tuple(w[i:i+1] for i in range(len(w)))
       if not tup:
           continue
       freqs[tup] = freqs.get(tup, 0) + 1
   return freqs


def _count_pairs(word_freq: Dict[Word, int]) -> Dict[Pair, int]:
   """Count adjacent token pairs across the corpus, weighted by word frequency."""
   pair_counts: Dict[Pair, int] = {}
   for word, count in word_freq.items(): # (b'h',b'e',b'l',b'l',b'o'): 2
       if len(word) < 2:
           continue
       # Count unique adjacent pairs occurrences in this word weighted by frequency
       # For BPE, we count each adjacent pair occurrence times the word frequency.
       for i in range(len(word) - 1):
           pair = (word[i], word[i + 1]) # (b'h', b'e')
           pair_counts[pair] = pair_counts.get(pair, 0) + count
   return pair_counts


def _merge_pair_in_word(word: Word, pair: Pair) -> Word:
   """Return a new word with all occurrences of the given pair merged."""
   a, b = pair
   merged: List[bytes] = []
   i = 0
   n = len(word)
   # E.g. b'a', b'b', b'c', b'd' and b'ab' becomes b'ab', b'c', b'd'
   while i < n:
       if i < n - 1 and word[i] == a and word[i + 1] == b: # match this new token
           merged.append(a + b)
           i += 2
       else:
           merged.append(word[i])
           i += 1
   return tuple(merged)


def _apply_merge(word_freq: Dict[Word, int], pair: Pair) -> Dict[Word, int]:
   """Apply a merge to all words. Return updated word_freq with merged words combined."""
   new_freq: Dict[Word, int] = {}
   for word, count in word_freq.items():
       new_w = _merge_pair_in_word(word, pair) # (b'h',b'e',b'l',b'l',b'o'), (b'h, b'e')
       new_freq[new_w] = new_freq.get(new_w, 0) + count
   return new_freq


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Pair]]:
   """
   Train a byte-level BPE tokenizer.


   Args:
   input_path: path to training text file (bytes are used directly)
   vocab_size: maximum size of final vocab including 256 bytes and specials
   special_tokens: list of strings to include; they do NOT affect merges


   Returns:
   vocab: dict mapping token id -> token bytes
   merges: list of (token1_bytes, token2_bytes) in creation order
   """
   if not os.path.isfile(input_path):
       raise FileNotFoundError(input_path)
   if vocab_size <= 0:
       raise ValueError("vocab_size must be positive")


   # Base vocabulary size: 256 bytes + specials
   base = 256 + len(special_tokens)
   if vocab_size < base:
       raise ValueError("vocab_size too small for base vocab (256 bytes + specials)")


   # Our output
   vocab: Dict[int, bytes] = {}
   merges: List[Pair] = []


   # Step1: Vocabulary initialization
   # 256 single-byte tokens in ascending order
   idx = 0
   for b in range(256):
       vocab[idx] = bytes([b])
       idx += 1


   # Step2: Pre-tokenization
   # Set desired_num_chunks > 1 to enable parallel chunked counting aligned to a special token.
   # For example: desired_num_chunks=4 and split_special_token=b"<|endoftext|>".
   use_chunked = True
   if use_chunked:
       word_freq = _read_and_pretokenize_chunked(
           input_path,
           desired_num_chunks=os.cpu_count(),
           split_special_token=b"<|endoftext|>",
           workers=None,
       )
   else:
       word_freq = _read_and_pretokenize(input_path)


   # E.g. {(b'h',b'e',b'l',b'l',b'o'): 2, (b'w',b'o',b'r',b'l',b'd'): 1}


   # Step3: Merge
   created_tokens: List[bytes] = []  # merged token bytes in creation order


   # Iteratively merge until reaching target size or no more pairs
   target_new_tokens = vocab_size - base


   while len(created_tokens) < target_new_tokens:
       pair_counts = _count_pairs(word_freq) # [(b'w', b'o'): 10, ...]
       if not pair_counts:
           break
       # Select best pair by highest count, then lexicographically greater pair on tie
       # max over items: key is (count, pair)
       best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
       # Apply merge
       word_freq = _apply_merge(word_freq, best_pair)
       # Record
       merges.append(best_pair)
       merged_token = best_pair[0] + best_pair[1] # b'at'
       created_tokens.append(merged_token)


   # Build final vocab id -> bytes
   # Special tokens first, in given order, encoded as utf-8
   for tok in special_tokens:
       vocab[idx] = tok.encode("utf-8")
       idx += 1
   # Then merged tokens in creation order (truncate if needed)
   for mt in created_tokens[: max(0, vocab_size - idx)]:
       vocab[idx] = mt
       idx += 1


   # If we exited early due to no pairs, idx may be < vocab_size; that is acceptable
   return vocab, merges


if __name__ == "__main__":
   # import tempfile
   #
   # corpus = (
   #     b"low low low low low\n"
   #     b"lower lower widest widest widest\n"
   #     b"newest newest newest newest newest newest\n"
   # )
   # with tempfile.NamedTemporaryFile(delete=False) as tmp:
   #     tmp.write(corpus)
   # path = tmp.name
   start = time.time()
   path = "../data/TinyStoriesV2-GPT4-valid.txt"
   vocab, merges = train_bpe(path, vocab_size=10000, special_tokens=["<|endoftext|>"])
   print("Merges (in order):")
   for a, b in merges:
       print((a.decode('latin1'), b.decode('latin1')))
   print("Total vocab size:", len(vocab))
   print(time.time() - start)