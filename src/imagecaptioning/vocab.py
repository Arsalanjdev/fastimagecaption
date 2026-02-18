from collections import Counter
from typing import List
import json


class Vocabulary:
    """
    Vocabulary object that maps words to indices and tracks token
    frequencies.

    The vocabulary is initialized with four special tokens:
    <PAD>, <UNK>, <SOS>, and <EOS>. Words are added based on a frequency
    threshold: words whose frequency is below the threshold are mapped
    to the <UNK> token. The class maintains bidirectional mappings
    (word2idx and idx2word) and keeps a frequency counter for tokens.

    Args:
        tokens (List[str] | None): Optional list of caption strings used
            to build the vocabulary.
        threshold (int): Minimum frequency required for a word to be
            assigned its own index (otherwise mapped to <UNK>).
    """

    def __init__(self, tokens: List[str] | None = None, threshold: int = 2):
        """
        Initialize the vocabulary, optionally building it from provided
        caption tokens.
        """
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.idx = 4
        self.threshold = threshold
        self.freq = Counter(" ".join(tokens).split()) if tokens else Counter()

        if tokens:
            for caption in tokens:
                for word in caption.split():
                    self.add_word(word)

    def add_word(self, word: str):
        """
        Add a word to the vocabulary according to the frequency threshold.

        If the word frequency is below the threshold, the word is mapped
        to <UNK>. Otherwise, the word is assigned a new index if it is not
        already present in the vocabulary.

        Args:
            word (str): Word to add to the vocabulary.
        """
        if self.freq[word] < self.threshold:
            word = "<UNK>"
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            self.freq[word] += 1

    def __len__(self):
        """
        Return the number of tokens currently stored in the vocabulary.

        Returns:
            int: Size of the vocabulary (including special tokens).
        """
        return len(self.word2idx)

    def dump_to_json(self, filepath: str) -> None:
        """
        Serialize the vocabulary to a JSON file.

        Exports the word-to-index mapping (word2idx), index-to-word mapping
        (idx2word), vocabulary size, and special token definitions (<PAD>, <SOS>,
        <EOS>, <UNK>) to a JSON file for later use in inference and model
        deployment.

        Args:
            filepath (str): Path where the JSON file will be saved.

        Returns:
            None
        """
        vocab_data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "vocab_size": len(self),
            "pad_token": "<PAD>",
            "start_token": "<SOS>",
            "end_token": "<EOS>",
            "unknown_token": "<UNK>",
        }
        with open(filepath, "w") as f:
            json.dump(vocab_data, f, indent=2)
