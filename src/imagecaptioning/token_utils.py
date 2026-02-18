import re
from typing import List


def tokenize(text: str) -> List[str]:
    """
    Tokenize a text string into lowercase word tokens.

    This function extracts alphanumeric word tokens using a regular
    expression and converts the input text to lowercase before
    tokenization.

    Args:
        text (str): Input text to tokenize.

    Returns:
        List[str]: A list of extracted lowercase word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def split_imagefile_captions(captions_filepath: str) -> List[List[str]]:
    """
    Read an image-caption file and split it into image–caption pairs.

    The function expects a CSV-like file where each line (after the
    header) contains an image filename followed by a caption separated
    by the first comma. The header line and the final empty line are
    skipped.

    Args:
        captions_filepath (str): Path to the captions file.

    Returns:
        List[List[str]]: A list of [image_filename, caption] pairs.
    """
    with open(captions_filepath, "r") as f:
        text = f.read()
    lines = text.split("\n")[1:-1]  # skip header and empty line
    return [line.split(",", 1) for line in lines]
