from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import numpy as np
from PIL import Image
import io
import torch
from torchvision import models, transforms
import torch.nn as nn

app = FastAPI(
    title="Image Embedding & Classification Service",
    description="Convert images to vector embeddings and classify content"
)

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()  # Set to evaluation mode

# Load ImageNet class labels (for classification)
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_embedding_and_class(image: Image.Image) -> Dict:
    """Convert PIL Image to embedding vector and classify content"""
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        # Get embedding (features before final layer)
        features = nn.Sequential(*list(model.children())[:-1])(image_tensor)
        embedding = features.squeeze().cpu().numpy().tolist()
        
        # Get classification
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        top5_probs, top5_classes = torch.topk(probs, 5)
        
        # Format classification results
        classification = [
            {
                "class": classes[top5_classes[i]],
                "confidence": float(top5_probs[i])
            }
            for i in range(5)
        ]
    
    return {
        "embedding": embedding,
        "classification": classification
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image - returns embedding and classification"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Get embedding and classification
        results = get_embedding_and_class(image)
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)