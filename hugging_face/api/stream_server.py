from fastapi import FastAPI, WebSocket
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading
import asyncio

app = FastAPI()

# Load tokenizer & model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 1. Receive prompt from client
            prompt = await websocket.receive_text()

            # 2. Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )

            # 3. Prepare streaming generator
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=False
            )

            # 4. Run model.generate() in background thread
            def generate():
                model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    streamer=streamer,
                )

            generation_thread = threading.Thread(target=generate)
            generation_thread.start()

            # 5. Send tokens as they stream in
            for token in streamer:
                await websocket.send_text(token)
                await asyncio.sleep(0.03)

            await websocket.send_text("[END]")  # optional signal to client
    except Exception as e:
        await websocket.close()
        print(f"WebSocket connection closed: {e}")
