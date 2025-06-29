# Image Embedding & Classification Service

A FastAPI service that converts images to vector embeddings and provides ImageNet classification using ResNet-50.

## Quick Start

```bash
pip install -r requirements.txt
./run.sh
```

Service runs at `http://localhost:8080`

## API

### POST `/analyze`
Upload an image to get embeddings and classification.

**Response:**
```json
{
  "embedding": [2048 float values],
  "classification": [
    {"class": "golden retriever", "confidence": 85.23},
    {"class": "Labrador retriever", "confidence": 12.45}
  ]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8080/analyze" -F "file=@image.jpg"
```

### GET `/health`
Health check endpoint.

## Features

- 2048-dimensional embeddings from ResNet-50
- Top-5 ImageNet classification with confidence scores
- Supports JPEG, PNG, and other common formats
- Interactive docs at `/docs`

## Dependencies

- FastAPI, PyTorch, torchvision, Pillow
- Pre-trained ResNet-50 model
- ImageNet class labels (1000 classes)
