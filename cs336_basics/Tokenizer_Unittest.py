# test_tokenizer.py
# Unittest suite for your BPE Tokenizer.
# Assumes your implementation is in tokenizer.py exposing:
#   - class Tokenizer
#   - function _token_str_to_bytes (used by from_files)
#
# Run:
#   python -m unittest -v test_tokenizer.py

import os
import json
import unittest
import tempfile
from pathlib import Path

# Adjust import as needed if your module name is different
from Tokenizer import Tokenizer, _token_str_to_bytes


def write_example_files(tmpdir):
    """
    Create vocab.json and merges.txt that match the slide example:

      vocab = {
        0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t",
        6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"
      }
      merges = [
        (b"t", b"h"),
        (b" ", b"c"),
        (b" ", b"a"),
        (b"th", b"e"),
        (b" a", b"t"),
      ]

    File formats expected by your from_files:
      - vocab.json: { "token": id, ... }
      - merges.txt: one pair per line: "A B", comments start with '#'
    """
    vocab_obj = {
        " ": 0,
        "a": 1,
        "c": 2,
        "e": 3,
        "h": 4,
        "t": 5,
        "th": 6,
        "Ġc": 7,
        "Ġa": 8,
        "the": 9,
        "Ġat": 10
    }
    vocab_path = os.path.join(tmpdir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_obj, f, ensure_ascii=False)

    merges_lines = [
        "t h",  # merge t + h → 'th'
        "Ġ c",  # merge space + c
        "Ġ a",  # merge space + a
        "th e",  # merge th + e → 'the'
        "Ġa t",  # merge ' a' + 't' → ' at'
    ]
    merges_path = os.path.join(tmpdir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for line in merges_lines:
            f.write(line + "\n")

    return vocab_path, merges_path


class TestTokenizer(unittest.TestCase):

    def test__token_str_to_bytes_helper(self):
        # Sanity check for helper with leading spaces and normal tokens
        s1 = " c"
        s1_1 = "Ġc"
        s2 = "a"
        b1 = _token_str_to_bytes(s1)
        b1_1 = _token_str_to_bytes(s1_1)
        b2 = _token_str_to_bytes(s2)
        self.assertEqual(b1, b" c")
        self.assertEqual(b1_1, b" c")
        self.assertEqual(b2, b"a")

    def test_from_files_and_slide_example(self):
        # Setup files
        with tempfile.TemporaryDirectory() as td:
            vocab_path, merges_path = write_example_files(td)

            # Build tokenizer from files
            tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=None)

            self.assertIn(b" c", tok.bytes_to_id)
            self.assertIn(b" a", tok.bytes_to_id)
            self.assertIn(b" at", tok.bytes_to_id)
            self.assertIn(b"the", tok.bytes_to_id)
            self.assertEqual(tok.bytes_to_id[b" c"], 7)
            self.assertEqual(tok.bytes_to_id[b" at"], 10)

            self.assertTrue(all(isinstance(p[0], bytes) and isinstance(p[1], bytes) for p in tok.merges))
            self.assertGreaterEqual(len(tok.merges), 5)

    def test_from_files_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            vocab_path, merges_path = write_example_files(td)

            # build tokenizer and test encode/decode
            tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=None)

            text = "the cat ate"
            ids = tok.encode(text)
            self.assertEqual(ids, [9, 7, 1, 5, 10, 3])

            decoded = tok.decode(ids)
            self.assertEqual(decoded, text)


    def test_from_files_and_slide_example_encode(self):
        with tempfile.TemporaryDirectory() as td:

            vocab_path, merges_path = write_example_files(td)

            # build tokenizer and test encode/decode
            tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=None)

            text = "the cat ate"
            ids = tok.encode(text)
            self.assertEqual(ids, [9, 7, 1, 5, 10, 3])

            # roundtrip = tok.decode(ids)
            # self.assertEqual(roundtrip, text)
    # def test_from_files_and_slide_example_encode_iter(self):
    #     with tempfile.TemporaryDirectory() as td:
    #
    #         vocab_path, merges_path = write_example_files(td)
    #
    #         # build tokenizer and test encode/decode
    #         tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=None)
    #
    #         whole = "the cat ate"
    #         ids_whole = tok.encode(whole)
    #
    #         # simulate streaming: two chunks, no cross-chunk merges
    #         chunks = ["the ", "cat ate"]
    #         ids_stream = list(tok.encode_iterable(chunks))
    #
    #         self.assertEqual(ids_stream, ids_whole)
    #         self.assertEqual(tok.decode(ids_stream), whole)
if __name__ == "__main__":
    unittest.main(verbosity=2)
