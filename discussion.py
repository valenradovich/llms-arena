from fastapi import FastAPI
from mlx_lm import load, stream_generate
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

app = FastAPI()

# loading both models globally
llama_model, llama_tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-bf16")
qwen_model, qwen_tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-bf16")

class DiscussionRequest(BaseModel):
    topic: str
    num_turns: int = 3  # number of back-and-forth exchanges
    system_prompt: str = """You are participating in a thoughtful discussion. 
Respond to the previous message and add your perspective."""

async def generate_response(model, tokenizer, prompt: str, system_prompt: str, context: str = "", turn_number: int = 0, model_name: str = ""):
    # Dynamic stances based on the topic
    stance_prompts = {
        "Llama": """You are Llama, debating against Qwen.
You are a strong proponent of the positive aspects regarding this topic.
Your role is to:
- Advocate for the benefits and opportunities
- Present optimistic yet well-reasoned arguments
- Counter negative perspectives with historical examples and data
- Keep responses focused and under 100 words
Your goal is to convince Qwen of the positive potential.""",
        
        "Qwen": """You are Qwen, debating against Llama.
You are thoughtfully critical and cautious about this topic.
Your role is to:
- Highlight potential risks and challenges
- Present pragmatic concerns and limitations
- Question optimistic assumptions with real-world examples
- Keep responses focused and under 100 words
Your goal is to convince Llama to consider the downsides."""
    }

    messages = [
        {"role": "system", "content": f"""{stance_prompts[model_name]}
When you see "Turn X - Llama:" in the conversation, that's Llama's optimistic perspective.
When you see "Turn X - Qwen:" in the conversation, that's Qwen's critical perspective.
Consider the ENTIRE conversation history when forming your response."""},
        {"role": "user", "content": f"""Turn {turn_number + 1}

Full conversation history:
{context}

Topic: {prompt}

Provide your perspective, addressing points from the entire discussion above."""}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    response_text = ""
    async for response in async_stream_generate(model, tokenizer, formatted_prompt, max_tokens=512):
        response_text += response.text
    
    # Clean up the response by removing any "Llama:" or "Qwen:" prefixes
    cleaned_response = response_text
    for prefix in ["Llama:", "Qwen:"]:
        cleaned_response = cleaned_response.replace(prefix, "").strip()
    
    return cleaned_response

# Wrapper to convert a synchronous generator to an asynchronous one
async def async_stream_generate(model, tokenizer, formatted_prompt, max_tokens):
    for response in stream_generate(model, tokenizer, formatted_prompt, max_tokens):
        yield response

@app.post("/discuss")
async def generate_discussion(request: DiscussionRequest):
    async def discussion_generator():
        try:
            conversation = []
            context = ""
            
            yield f"Topic: {request.topic}\n\n"
            
            for turn in range(request.num_turns):
                # llama's turn
                llama_response = await generate_response(
                    llama_model, 
                    llama_tokenizer, 
                    request.topic, 
                    request.system_prompt, 
                    context,
                    turn,
                    "Llama"
                )
                conversation.append(f"Turn {turn + 1} - Llama: {llama_response}")
                context = "\n\n".join(conversation)
                yield f"Llama: {llama_response}\n\n"
                
                # qwen's turn
                qwen_response = await generate_response(
                    qwen_model, 
                    qwen_tokenizer, 
                    request.topic, 
                    request.system_prompt, 
                    context,
                    turn,
                    "Qwen"
                )
                conversation.append(f"Turn {turn + 1} - Qwen: {qwen_response}")
                context = "\n\n".join(conversation)
                yield f"Qwen: {qwen_response}\n\n"
        except Exception as e:
            yield f"Error during generation: {str(e)}\n"

    return StreamingResponse(
        discussion_generator(),
        media_type="text/plain",
    ) 