from fastapi import FastAPI
from mlx_lm import load, stream_generate
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel

app = FastAPI()

# loading model globally to reuse across requests
model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")

async def generate_stream(prompt: str, system_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    for response in stream_generate(model, tokenizer, formatted_prompt, max_tokens=512):
        yield response.text

class PromptRequest(BaseModel):
    prompt: str
    system_prompt: str | None = "You are a helpful AI assistant that provides clear, accurate, and concise responses."

@app.post("/generate")
async def generate_text(request: PromptRequest):
    return StreamingResponse(
        generate_stream(request.prompt, request.system_prompt),
        media_type="text/event-stream"
    )