import multiprocessing
import regex as re
from collections import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries

'''
Problem (train_bpe): BPE Tokenizer Training (15 points)

Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
tokenizer. Your BPE training function should handle (at least) the following input parameters:

input_path: str Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:

vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
    lary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
'''
def pretokenize(chunk: str, special_tokens: list[str], output_queue: multiprocessing.Queue):
    escaped_special_tokens = [re.escape(special_token) for special_token in special_tokens]
    pattern = '|'.join(escaped_special_tokens)
    non_capturing_pattern = '(?:' + pattern + ')'
    split_chunks = re.split(non_capturing_pattern, chunk)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = {}
    for split_chunk in split_chunks:
        matches = re.finditer(PAT, split_chunk)
        for match in matches:
            pretoken = match.group()
            if pretoken not in pretokens:
                pretokens[pretoken] = 1
            else:
                pretokens[pretoken] += 1
    output_queue.put(pretokens)

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str], 
    num_processes: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

        # multiprocessing implementation
        output_queue = multiprocessing.Queue()
        processes = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            process = multiprocessing.Process(target=pretokenize, args=(chunk, special_tokens, output_queue))
            process.start()
            processes.append(process)

        merged_pretokens = Counter()
        for process in processes:
            pretokens = output_queue.get()
            merged_pretokens += Counter(pretokens)
            process.join() # https://stackoverflow.com/questions/29810041/python-weird-behavior-with-multiprocessing-join-does-not-execute
        
        merged_pretokens = dict(merged_pretokens)
        byte_pretokens = {tuple([int.to_bytes() for int in pretoken.encode('utf-8')]): count for pretoken, count in merged_pretokens.items()}

        # initialize vocab
        vocab = {token_id: token_id.to_bytes() for token_id in range(256)}
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode('utf-8')
        
        # initialize merges
        merges = []

        while len(vocab) < vocab_size:
            byte_pair_counts = {}
            for token in byte_pretokens:
                for index in range(len(token)-1):
                    byte_pair = (token[index], token[index+1])
                    if byte_pair not in byte_pair_counts:
                        byte_pair_counts[byte_pair] = byte_pretokens[token]
                    else:
                        byte_pair_counts[byte_pair] += byte_pretokens[token]
                        
            max_count_byte_pair = max(sorted(byte_pair_counts, reverse=True), key=byte_pair_counts.get)

            # merging
            merges.append(max_count_byte_pair)
            vocab[len(vocab)] = max_count_byte_pair[0] + max_count_byte_pair[1]

            updated_byte_pretokens = {}
            # update pretokens by applying the merge
            for token in byte_pretokens:
                updated_token = token
                index = 0
                index_upper_bound = len(updated_token) - 1
                while index < index_upper_bound:
                    byte_pair = (updated_token[index], updated_token[index+1])
                    if byte_pair == max_count_byte_pair:
                        updated_token = updated_token[:index] + (byte_pair[0] + byte_pair[1], ) + updated_token[index+2:]
                        index_upper_bound -= 1
                    index += 1

                updated_byte_pretokens[updated_token] = byte_pretokens[token]

            byte_pretokens = updated_byte_pretokens

    return vocab, merges


if __name__ == '__main__':
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()


    TinyStories_train_set_path = 'cs336_basics/data/TinyStoriesV2-GPT4-train.txt'
    TinyStories_validation_set_path = 'cs336_basics/data/TinyStoriesV2-GPT4-valid.txt'
    vocab_size = 512
    special_tokens = ['<|endoftext|>', '<|examplespecialtoken|>']
    
    vocab, merges = train_bpe(TinyStories_validation_set_path, vocab_size, special_tokens, 1)


    profiler.disable()
    profiler.print_stats(sort='time')