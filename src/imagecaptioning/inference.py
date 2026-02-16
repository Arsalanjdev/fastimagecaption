import argparse
import json
import os
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.imagecaptioning.model import HybridModelAttention


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def load_vocab(vocab_json_path: str) -> Dict[str, Dict]:
    """
    Load vocabulary mappings from a JSON file.

    The JSON file is expected to contain:
        - word2idx: mapping from words to integer indices
        - idx2word: mapping from indices (string keys) to words

    Parameters
    ----------
    vocab_json_path : str
        Path to the vocabulary JSON file.

    Returns
    -------
    Dict[str, Dict]
        Dictionary containing:
            {
                "word2idx": Dict[str, int],
                "idx2word": Dict[int, str]
            }
    """
    with open(vocab_json_path, "r") as f:
        data = json.load(f)

    word2idx = data.get("word2idx", {})
    idx2word_raw = data.get("idx2word", {})
    idx2word = {int(k): v for k, v in idx2word_raw.items()}

    return {"word2idx": word2idx, "idx2word": idx2word}


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Load and preprocess an image for caption generation.

    Steps:
        1. Load image
        2. Convert to RGB
        3. Resize to (224, 224)
        4. Convert to tensor
        5. Add batch dimension
        6. Move to target device

    Parameters
    ----------
    image_path : str
        Path to the input image.
    device : torch.device
        Device where the tensor will be moved.

    Returns
    -------
    torch.Tensor
        Preprocessed image tensor of shape (1, 3, 224, 224).
    """
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return transform(img).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def generate_caption(
    image_path: str,
    model: HybridModelAttention,
    vocab: Dict,
    device: torch.device,
    max_len: int = 20
) -> str:
    """
    Generate an image caption using an attention-based hybrid model.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    model : HybridModelAttention
        Captioning model.
    vocab : Dict
        Vocabulary containing word2idx and idx2word mappings.
    device : torch.device
        Device for inference.
    max_len : int, optional
        Maximum caption length, by default 20.

    Returns
    -------
    str
        Generated caption.
    """
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]

    sos = word2idx.get("<SOS>", 2)
    eos = word2idx.get("<EOS>", 3)

    model.eval()
    img_tensor = preprocess_image(image_path, device)

    with torch.no_grad():
        encoder_out = model.encoder(img_tensor)
        h, c = model.decoder.init_hidden(encoder_out)

        current_id = torch.tensor([sos], dtype=torch.long, device=device)
        embedding = model.decoder.embedding(current_id)

        tokens = []

        for _ in range(max_len):
            context, _ = model.decoder.attention(encoder_out, h)
            inp = torch.cat([embedding, context], dim=1)

            h, c = model.decoder.lstm(inp, (h, c))
            logits = model.decoder.fc(h)

            next_id = int(logits.argmax(dim=1).item())
            if next_id == eos:
                break

            tokens.append(idx2word.get(next_id, "<UNK>"))
            current_id = torch.tensor([next_id], dtype=torch.long, device=device)
            embedding = model.decoder.embedding(current_id)

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point for generating captions from an image.
    """
    parser = argparse.ArgumentParser(description="Generate caption for an input image")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default="model_weights.pt", help="Path to .pt model file")
    parser.add_argument("--vocab", default="vocab.json", help="Path to vocab JSON file")
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint (.pth)")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu or cuda)")
    parser.add_argument("--max-len", type=int, default=20, help="Maximum caption length")

    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"Vocab file not found: {args.vocab}")

    vocab = load_vocab(args.vocab)
    vocab_size = len(vocab["word2idx"])

    state_dict = torch.load(args.model, map_location=device)

    model = HybridModelAttention(vocab_size).to(device)
    model.load_state_dict(state_dict)

    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            state = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state)
        else:
            print(f"Warning: checkpoint not found at {args.checkpoint}. Using random weights.")

    caption = generate_caption(args.image, model, vocab, device, max_len=args.max_len)
    print(caption)


if __name__ == "__main__":
    main()
