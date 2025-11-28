import os
import json
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ===== 配置区域 =====
BASE_MODEL_PATH = "./../../chatglm-6b"        # 原始 ChatGLM-6B 模型目录
CHECKPOINT_PATH = "./../../../output/chatglm-6b-pt-128-2e-2/checkpoint-3000"    # 你的 P-Tuning v2 微调权重目录
PRE_SEQ_LEN = 128                       # 你训练时用的 pre_seq_len，一定要对上

# ===== 加载模型 =====
print(">>> Loading config...")
config = AutoConfig.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    pre_seq_len=PRE_SEQ_LEN
)

print(">>> Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

print(">>> Loading base model...")
model = AutoModel.from_pretrained(
    BASE_MODEL_PATH,
    config=config,
    trust_remote_code=True
)

# 加载 prefix tuning 权重
print(">>> Loading prefix tuning checkpoint...")
prefix_state_dict = torch.load(
    os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"),
    map_location="cpu"
)

new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_key = k[len("transformer.prefix_encoder."):]
        new_prefix_state_dict[new_key] = v

model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
print(">>> Prefix encoder loaded.")

# 设备与精度设置
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
model = model.eval()

print(f">>> Model loaded on {device}.")

# ===== FastAPI 定义 =====
app = FastAPI(title="ChatGLM P-Tuning API", version="1.0.0")

class ChatRequest(BaseModel):
    query: str
    history: list | None = None
    max_length: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95

class ChatResponse(BaseModel):
    response: str
    history: list

# ===== 普通非流式接口 =====
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = req.history or []
    query = req.query

    response, history = model.chat(
        tokenizer,
        query,
        history=history,
        max_length=req.max_length,
        top_p=req.top_p,
        temperature=req.temperature
    )

    return ChatResponse(response=response, history=history)

# ===== 流式接口（SSE） =====
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    使用 Server-Sent Events（text/event-stream）做流式输出。
    前端可以用 EventSource 或自己解析流。
    """

    def generate():
        history = req.history or []
        query = req.query

        # ChatGLM 的 stream_chat 一般是这样用：
        # for response, history in model.stream_chat(...):
        #   ...
        # 具体以你的 chatglm 版本为准，如果没有这个函数，可以参考官方 repo 修改。

        for response, history in model.stream_chat(
            tokenizer,
            query,
            history=history,
            max_length=req.max_length,
            top_p=req.top_p,
            temperature=req.temperature
        ):
            # 这里我们每一小段增量作为 delta 返回，方便前端累加
            data = {
                "delta": response,
                "history": history
            }
            # SSE 协议格式：以 "data: " 开头，后面是 JSON，再加两个换行
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        # 结束标记（类似 OpenAI 的 [DONE]）
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
