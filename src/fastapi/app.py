from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from src.imagecaptioning.inference import generate_caption
from src.imagecaptioning.model import HybridModelAttention
import torch
import json
import io

app = FastAPI(title="FastImageCaptioning", summary="Generate a caption based on the image you upload.", version="0.1.0")

def get_model(vocab_size: int,model_weights: str = "model_weights.pt"):
    model = HybridModelAttention(vocab_size)
    state_dict = torch.load(model_weights)
    model.load_state_dict(state_dict)
    return model

@app.post("/upload"):
async def upload_image(file: UploadFile = File()):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open("vocab.json") as f:
            vocab = json.load(f)
        model = get_model(len(vocab), model_weights="model_weights.pt")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        caption = generate_caption(image=image, device=device, vocab=vocab, model=model)
        return {"caption": caption}
    except Exception as e:
        return HTTPException(400,str(e))
