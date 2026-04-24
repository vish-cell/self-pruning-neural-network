from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load("model_0.1.pth", weights_only=False).to(device)
model.eval()

# CIFAR-10 classes
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1).item()

    return {
        "class_id": pred,
        "class_name": classes[pred]
    }