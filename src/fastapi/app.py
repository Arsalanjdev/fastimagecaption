from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from fastapi.staticfiles import StaticFiles
from src.imagecaptioning.inference import generate_caption
from src.imagecaptioning.model import HybridModelAttention
import torch
import json
import io

app = FastAPI(
    title="FastImageCaptioning",
    summary="Generate a caption based on the image you upload.",
    version="0.1.0",
)
app.mount("/app", StaticFiles(directory="static", html=True), name="static")


def load_vocab(vocab_json_path: str):
    with open(vocab_json_path, "r") as f:
        data = json.load(f)

    word2idx = data.get("word2idx", {})
    idx2word_raw = data.get("idx2word", {})
    idx2word = {int(k): v for k, v in idx2word_raw.items()}

    return {"word2idx": word2idx, "idx2word": idx2word}


# 1. Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 2. Load vocab once
vocab = load_vocab("vocab.json")
vocab_size = len(vocab["word2idx"])

# 3. Load model once
model = HybridModelAttention(vocab_size).to(device)
state_dict = torch.load("model_weights.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()  # important!


@app.post("/upload")
async def upload_image(file: UploadFile = File()):
    try:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        vocab = load_vocab("vocab.json")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        caption = generate_caption(image=image, device=device, vocab=vocab, model=model)
        return {"caption": caption}
    except Exception as e:
        return HTTPException(400, str(e))
