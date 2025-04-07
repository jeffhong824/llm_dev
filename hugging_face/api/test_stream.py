import asyncio
import websockets

async def chat():
    uri = "ws://localhost:8000/chat" 
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello, my name is")
        while True:
            try:
                response = await websocket.recv()
                if response == "[END]":
                    break
                print(response, end="", flush=True)
            except websockets.exceptions.ConnectionClosed:
                print("\n[Connection closed by server]")
                break

asyncio.run(chat())
