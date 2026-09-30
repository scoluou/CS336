import regex as re
import pickle
from collections.abc import Iterable


class BPETokenizer:
    def __init__(self, vocab, merges, sp_tokens=None):
        super().__init__()
        self.vocab = vocab
        self.merges = merges
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.bytes_to_id: dict[bytes, int] = {}
        for token_id, token_bytes in self.vocab.items():
            self.bytes_to_id[token_bytes] = token_id

        self.pari_to_create_index: dict[tuple[bytes, bytes], int]= {}
        for i in range(len(merges)):
            self.pari_to_create_index[merges[i]] = i

        self.sp_tokens = sp_tokens
        self.has_sp_tokens = sp_tokens is not None
        if self.has_sp_tokens:
            self.sp_token_id: dict[str, int] = {}
            for sp_token in sp_tokens:
                self.sp_token_id[sp_token] = self.bytes_to_id[sp_token.encode('utf-8')]

    @classmethod
    def from_files(cls, file_path, sp_tokens=None):
        with open(file_path, 'rb') as f:
            saved = pickle.load(f)
            return BPETokenizer(saved['vocab'], saved['merges'], sp_tokens)

    def encode(self, text: str) -> list[int]:
        encoded_ids: list[int] = []

        if self.has_sp_tokens:
            # Split text into segments
            sorted_sp_tokens = sorted(self.sp_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(sp_token) for sp_token in sorted_sp_tokens)
            segments = re.split(f"({pattern})", text) if pattern else [text]
        else:
            segments = [text]

        for segment in segments:
            # If segment is special token, append special token id and continue
            if self.has_sp_tokens and segment in self.sp_tokens:
                encoded_ids.append(self.sp_token_id[segment])
                continue

            split_text = self.pattern.findall(segment)

            for word in split_text:
                pre_token: tuple[bytes, ...] = tuple(bytes([token]) for token in word.encode('utf-8', errors='replace'))
                merged_token: list[bytes] = []
                while True:
                    # Find candidate merge pairs with highest priority
                    merge_index = -1
                    current_merge_priority = 2**31 - 1
                    for i in range(len(pre_token) - 1):
                        merge_priority = self.pari_to_create_index.get((pre_token[i], pre_token[i + 1]), -1)
                        if merge_priority != -1 and merge_priority < current_merge_priority:
                            merge_index = i
                            current_merge_priority = merge_priority

                    # Merge with highest priority
                    if merge_index >= 0:
                        merged_token.extend(pre_token[:merge_index])
                        merged_token.append(pre_token[merge_index] + pre_token[merge_index + 1])
                        merged_token.extend(pre_token[merge_index+2:])

                        # print(f'{pre_token} -> {merged_token}')
                        pre_token = tuple(merged_token)
                        merged_token.clear()
                    else:
                        break
                encoded_ids.extend([self.bytes_to_id.get(token_bytes, 0) for token_bytes in pre_token])

        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # Convert to a full bytes and decode once rather than decode each byte
        decoded_bytes = [self.vocab[id] for id in ids]
        return b''.join(decoded_bytes).decode('utf-8', errors='replace')
