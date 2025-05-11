from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import pipeline
import torch
import os

app = FastAPI()

# Cache for loaded models
MODEL_CACHE = {}

class ModelRequest(BaseModel):
    model_id: str
    inputs: str
    task: str = "text-generation"
    parameters: Optional[Dict] = None

class ModelResponse(BaseModel):
    outputs: List[Dict]
    error: Optional[str] = None

@app.post('/inference', response_model=ModelResponse)
async def run_inference(request: ModelRequest):
    try:
        cache_key = f"{request.task}_{request.model_id}"
        
        if cache_key not in MODEL_CACHE:
            MODEL_CACHE[cache_key] = pipeline(
                task=request.task,
                model=request.model_id,
                device=0 if torch.cuda.is_available() else -1
            )
            
        model = MODEL_CACHE[cache_key]
        params = request.parameters or {}
        
        outputs = model(request.inputs, **params)
        return ModelResponse(outputs=outputs if isinstance(outputs, list) else [outputs])
    except Exception as e:
        return ModelResponse(outputs=[], error=str(e))

@app.get('/models')
async def list_models():
    return {"loaded_models": list(MODEL_CACHE.keys())}

@app.get('/health')
async def health_check():
    return {
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available()
    }