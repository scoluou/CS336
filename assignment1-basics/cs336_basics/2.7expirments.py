from tests.adapters import *
import numpy as np
import pathlib


def compute_compression_ratio():
    paragraph = """Once upon a time, there was a little girl called Lucy. Lucy loved the rain.
    One day, the sky started to turn purple, and before long, it was pouring. Lucy put on her raincoat and went outside to play.
    In her garden, she saw a big pile of hay. She ran up to it and jumped in, enjoying the feeling of the rain on her face.
    When Lucy's mom came out to see where her daughter had gone, she called out, "Lucy! Where are you? What are you doing?"
    Lucy poked her head up from the pile of hay and said, "I'm here! I'm playing in the rain!"
    Her mom smiled and said, "That sounds like a lot of fun! Just make sure you stay dry!"
    Lucy stayed in the hay, enjoying the sound of the rain. When she was done, she said goodbye to the rain, and went back inside.
    From then on, every time it rained, Lucy would go out and play in the hay, feeling the purple sky above her.
    <|endoftext|>"""
    tiny_story_tokenizer = BPETokenizer.from_files('../data/data/tinystories_vocab_merge.pkl', ["<|endoftext|>"])
    tokens = tiny_story_tokenizer.encode(paragraph)

    compression_ratio = len(paragraph.encode('utf-8')) / (len(tokens))
    print(compression_ratio)


def compute_throughput():
    tiny_story_tokenizer = BPETokenizer.from_files('../data/data/tinystories_vocab_merge.pkl', ["<|endoftext|>"])

    path = '../data/TinyStoriesV2-GPT4-valid.txt'
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        file_size = pathlib.Path(path).stat().st_size
        start_time = time.perf_counter()
        for id in tiny_story_tokenizer.encode_iterable(f):
            ...
        end_time = time.perf_counter()

        bytes_per_second = file_size / (end_time - start_time)

        print(f'bytes/second {bytes_per_second:_.1f}')

        time_to_encode_Pile_dataset = 825e9 / bytes_per_second
        print(f'time to encode Pile dataset {time_to_encode_Pile_dataset / 3600:.2f}h')


def encode_tiny_stories():
    tiny_story_tokenizer = BPETokenizer.from_files('../data/data/tinystories_vocab_merge.pkl', ["<|endoftext|>"])

    train_file = '../data/TinyStoriesV2-GPT4-train.txt'
    valid_file = '../data/TinyStoriesV2-GPT4-valid.txt'

    with open(train_file, 'r', encoding='utf-8', errors='replace') as f:
        train_tokens = np.fromiter(tiny_story_tokenizer.encode_iterable(f), dtype=np.uint16)
        np.save('../data/TinyStories_train', train_tokens)

    with open(valid_file, 'r', encoding='utf-8', errors='replace') as f:
        valid_tokens = np.fromiter(tiny_story_tokenizer.encode_iterable(f), dtype=np.uint16)
        np.save('../data/TinyStories_valid', valid_tokens)





if __name__ == '__main__':
    ...

    compute_compression_ratio()

    compute_throughput()

    # encode_tiny_stories()
