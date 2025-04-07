from fastapi import FastAPI, WebSocket
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading
import asyncio

app = FastAPI()

# 模型快取池（避免每次重新下載）
model_cache = {}

def load_model_and_tokenizer(model_name: str):
    if model_name in model_cache:
        return model_cache[model_name]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        return tokenizer, model

    except Exception as e:
        raise RuntimeError(f"模型加載失敗: {str(e)}")


@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 1. Client 傳來一段 JSON 格式資料
            client_data = await websocket.receive_json()
            prompt = client_data.get("prompt", "").strip()
            model_name = client_data.get("model", "gpt2").strip()

            if not prompt:
                await websocket.send_text("[ERROR] Prompt 不可為空。")
                continue

            try:
                tokenizer, model = load_model_and_tokenizer(model_name)
            except RuntimeError as e:
                await websocket.send_text(f"[ERROR] 無法載入模型 {model_name}：{str(e)}")
                continue

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=False
            )

            def generate():
                model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    streamer=streamer,
                )

            generation_thread = threading.Thread(target=generate)
            generation_thread.start()

            for token in streamer:
                await websocket.send_text(token)
                await asyncio.sleep(0.03)

            await websocket.send_text("[END]")

    except Exception as e:
        await websocket.close()
        print(f"[Server] WebSocket connection closed due to: {e}")
