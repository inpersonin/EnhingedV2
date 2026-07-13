from __future__ import annotations
import os
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config import ONNX_MODEL_PATH
from inference_onnx import generate_response, get_loaded_checkpoint_path, is_model_loaded, load_model, unload_model
app = FastAPI(title='Enhinged V2 API', version='2.0.0')
DEFAULT_API_CHECKPOINT = os.getenv('ENHINGED_CKPT_PATH', ONNX_MODEL_PATH)
CORS_ORIGINS = [origin.strip() for origin in os.getenv('ENHINGED_CORS_ORIGINS', '*').split(',') if origin.strip()]
STARTUP_ERROR: Optional[str] = None
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS or ['*'], allow_credentials=False, allow_methods=['*'], allow_headers=['*'])

class ChatTurn(BaseModel):
    role: Literal['user', 'assistant', 'bot'] = 'user'
    content: str

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=110, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.01, le=5.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=5.0)
    do_sample: bool = True
    seed: Optional[int] = Field(default=None, ge=0)
    conversation_history: list[ChatTurn] = Field(default_factory=list)

class LoadRequest(BaseModel):
    checkpoint_path: str = DEFAULT_API_CHECKPOINT

def _ensure_model_loaded() -> None:
    if is_model_loaded():
        return
    load_model(DEFAULT_API_CHECKPOINT)

@app.on_event('startup')
def startup() -> None:
    global STARTUP_ERROR
    try:
        _ensure_model_loaded()
    except Exception as exc:
        STARTUP_ERROR = str(exc)

@app.on_event('shutdown')
def shutdown() -> None:
    unload_model()

@app.get('/')
def root() -> dict:
    return {'service': 'Enhinged V2 API', 'version': '2.0.0', 'model_loaded': is_model_loaded()}

@app.get('/health')
def health() -> dict:
    return {'status': 'ok', 'model_loaded': is_model_loaded(), 'checkpoint_path': get_loaded_checkpoint_path(), 'startup_error': STARTUP_ERROR}

@app.post('/generate')
def generate(payload: GenerateRequest) -> dict:
    try:
        _ensure_model_loaded()
        response = generate_response(prompt=payload.prompt, max_new_tokens=payload.max_new_tokens, temperature=payload.temperature, top_k=payload.top_k, top_p=payload.top_p, repetition_penalty=payload.repetition_penalty, do_sample=payload.do_sample, seed=payload.seed, conversation_history=[turn.model_dump() for turn in payload.conversation_history] or None)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {'response': response}

@app.post('/load')
def reload_model(payload: LoadRequest) -> dict:
    global STARTUP_ERROR
    try:
        load_model(payload.checkpoint_path)
        STARTUP_ERROR = None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {'status': 'loaded', 'checkpoint_path': get_loaded_checkpoint_path()}

@app.post('/unload')
def unload() -> dict:
    unload_model()
    return {'status': 'unloaded'}
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api:app', host='0.0.0.0', port=7860, reload=False)