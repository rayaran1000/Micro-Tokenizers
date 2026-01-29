"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from base import Tokenizer,get_stats,merge

class BPTokenizer(Tokenizer):
    '''
    Byte Pair tokenizer
    '''

    def __init__(self) -> None:
        super.__init__()
    
    def train(self,text,vocab_size,verbose=False):
        # Asserting that the vocab size minimum value is 256, since each individual char has a byte representation mandatory
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        tokens = list(map(int,text.encode("utf-8"))) # Converting byte encodings to int tokens
        merges = {} # Dictionary for storing the merges done by tokenizer : (int, int) -> int
        vocab = {idx : bytes([idx]) for idx in range(256)} # 0 to 255 reserved for individual characters

        for i in num_merges:
            # Get pair count values
            token_pair_counts = get_stats(tokens)
            # Pick the pair with the max count value
            merge_pair = max(token_pair_counts,key=token_pair_counts.get)
            # Increase the idx by 1, since the new merged token will take the next idx
            idx = 256 + 1
            # Merge the pair and replace with new idx
            tokens = merge(tokens,merge_pair,idx)
            # Add the merge pair and idx in merged dict
            merges[merge_pair] = idx
            # Add in vocabulary the new merged pair byte info, by fetching individual byte values and concatenate them
            vocab[idx] = vocab[merge_pair[0]] + vocab[merge_pair[1]]
            
            # Print if verbose is set to True
            if verbose:
                print(f"merged ({merge_pair[0]} , {merge_pair[1]}) to index position -> {idx}")
        
        # Save the new merge and vocab dict to class variables to be used in other functions within the class
        self.merges = merges
        self.vocab = vocab
    
    def encode(self,text):
        text_ids = list(text.encode("utf-8"))
        while len(text_ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(text_ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # The pair which was merged during training at the start will be considered as priority than the later ones, because after merging the new idx that was generated, could also have been used for merging in later cases, hence min is used.
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def decode(self,tokens):
        text_bytes =  b"".join(self.vocab[idx] for idx in tokens)
        text = text_bytes.decode("utf-8",errors="replace")
        return text
