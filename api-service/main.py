import os
import re
import json
import uuid
import asyncio
import subprocess
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal

app = FastAPI()

# ----------------- Paths -----------------
MODEL_CONFIG = os.getenv("MODEL_CONFIG")

# ----------------- CORS Middleware -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Input Schema -----------------
class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False

# ----------------- Prompt Formatter --------------
def build_prompt(messages: List[ChatMessage]) -> str:
    system_msg = next(
        (msg.content for msg in messages if msg.role == "system"),
        "You are a helpful assistant."
    )
    last_user_msg = next(
        (msg.content for msg in reversed(messages) if msg.role == "user"),
        None
    )

    if not last_user_msg:
        raise ValueError("No user message found in the request.")

    return f"<|system|>{system_msg}</s><|user|>{last_user_msg}</s><|assistant|>"

# ----------------- Tokenizer ---------------------
def tokenize_stream(text: str):
    word = ""
    for char in text:
        word += char
        if char in [' ', '\n']:
            yield word
            word = ""
    if word:
        yield word

# ----------------- Ignore case -----------------
CONTROL_TOKEN_RE = re.compile(
    r"(?:</?s>|<\|?(?:system|user|assistant|end|bos|eos)\|?>)",
    flags=re.IGNORECASE
)

# ----------------- Clean response -----------------
def clean_model_output(text: str) -> str:
    # Remove framework/control tokens
    text = CONTROL_TOKEN_RE.sub("", text)
    # If multiple assistant turns leaked, keep the first span only
    text = text.split("<|assistant|>", 1)[0]
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()
    
# ----------------- Extract Response -----------------
def extract_response(stdout: str):
    if "[BEGIN]:" not in stdout:
        cleaned = clean_model_output(stdout)
        return (cleaned if cleaned else "[No output from model]"), "fallback"

    after_begin = stdout.split("[BEGIN]:", 1)[1]
    if "[END]" in after_begin:
        raw = after_begin.split("[END]", 1)[0]
        return clean_model_output(raw), "stop"
    elif "[ABORT]" in after_begin:
        raw = after_begin.split("[ABORT]", 1)[0]
        return clean_model_output(raw), "length"
    else:
        return clean_model_output(after_begin), "error"

# ----------------- Genie Stream Generator -----------------
async def genie_stream_generator(prompt: str):
    cmd = ["genie-t2t-run", "-c", MODEL_CONFIG, "-p", prompt]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )

    buf = ""
    capturing = False

    while True:
        chunk = await process.stdout.read(64)  # read a bit more than 1 byte
        if not chunk:
            break

        buf += chunk.decode("utf-8", errors="ignore")

        if not capturing and "[BEGIN]:" in buf:
            buf = buf.split("[BEGIN]:", 1)[1]
            capturing = True

        if not capturing:
            continue

        # Stop conditions
        if "[END]" in buf or "[ABORT]" in buf:
            stop_tag = "[END]" if "[END]" in buf else "[ABORT]"
            text_to_stream, _ = buf.split(stop_tag, 1)

            text_to_stream = clean_model_output(text_to_stream)
            for token in tokenize_stream(text_to_stream):
                if token:
                    yield f"data: {json.dumps({'id': str(uuid.uuid4()), 'object':'chat.completion.chunk','choices':[{'delta':{'content': token},'index':0,'finish_reason':None}]})}\n\n"

            finish_reason = "stop" if stop_tag == "[END]" else "length"
            yield f"data: {json.dumps({'id': str(uuid.uuid4()), 'object':'chat.completion.chunk','choices':[{'delta':{},'index':0,'finish_reason':finish_reason}]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Stream partial tokens continuously
        cleaned = clean_model_output(buf)
        # Only stream complete tokens; keep the leftover partial in buf
        last_break = max(cleaned.rfind(" "), cleaned.rfind("\n"))
        if last_break >= 0:
            emit, leftover = cleaned[:last_break+1], cleaned[last_break+1:]
            for token in tokenize_stream(emit):
                if token:
                    yield f"data: {json.dumps({'id': str(uuid.uuid4()), 'object':'chat.completion.chunk','choices':[{'delta':{'content': token},'index':0,'finish_reason':None}]})}\n\n"
            # Reconstruct buf from the leftover plus any not-yet-cleaned tail
            # Find the same leftover in the original buf to avoid losing markers
            buf = leftover

    # EOF fallback
    if capturing and buf.strip():
        for token in tokenize_stream(clean_model_output(buf.strip())):
            if token:
                yield f"data: {json.dumps({'id': str(uuid.uuid4()), 'object':'chat.completion.chunk','choices':[{'delta':{'content': token},'index':0,'finish_reason':None}]})}\n\n"

    yield f"data: {json.dumps({'id': str(uuid.uuid4()), 'object':'chat.completion.chunk','choices':[{'delta':{},'index':0,'finish_reason':'error'}]})}\n\n"
    yield "data: [DONE]\n\n"
# ----------------- Completion Endpoint ------------
@app.post("/chat/completions")
async def chat_completion(request: Request, chat_request: ChatRequest):
    try:
        prompt = build_prompt(chat_request.messages)

        if len(prompt.strip()) < 30:
            return JSONResponse(
                status_code=400,
                content={"error": "Prompt too short to generate a meaningful response."}
            )

        # ---- Streaming ----
        if chat_request.stream:
            return StreamingResponse(
                genie_stream_generator(prompt),
                media_type="text/event-stream"
            )

        # ---- Non-Streaming ----
        result = subprocess.run(
            ["genie-t2t-run", "-c", MODEL_CONFIG, "-p", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False
        )

        output, finish_reason = extract_response(result.stdout)

        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "model": chat_request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output},
                    "finish_reason": finish_reason
                }
            ]
        }

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "model": chat_request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"[ERROR] {str(e)}"},
                        "finish_reason": "error"
                    }
                ]
            }
        )

# ----------------- Models Endpoint -----------------
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": os.getenv("MODEL_NAME"),
                "object": "model",
                "owned_by": "user",
                "size": "1B",
                "modified": time.strftime('%Y-%m-%dT%H:%M:%S')
            }
        ]
    }
