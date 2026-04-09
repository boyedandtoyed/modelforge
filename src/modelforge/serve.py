"""vLLM-compatible serving interface. Production: replace with vllm.AsyncLLMEngine."""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ModelForge Serve", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list[str] = []


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


@app.get("/health")
def health():
    return {"status": "ok", "note": "Demo mode — no model loaded. Use vllm.AsyncLLMEngine in production."}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Production with vLLM:
        from vllm import LLM, SamplingParams
        llm = LLM(model=adapter_path)
        outputs = llm.generate([req.prompt], SamplingParams(...))
    """
    demo_text = f"[Demo response to: {req.prompt[:60]}...] In production, load a fine-tuned adapter with vLLM."
    return GenerateResponse(
        text=demo_text,
        prompt_tokens=len(req.prompt.split()),
        completion_tokens=len(demo_text.split()),
        finish_reason="stop",
    )
