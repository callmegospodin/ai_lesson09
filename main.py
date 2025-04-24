import os
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openrouter import OpenRouter
from dotenv import load_dotenv
from typing import AsyncGenerator
import asyncio

# Завантажуємо змінні оточення
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Налаштування OpenRouter
llm = OpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",  # Можна змінити на будь-яку модель OpenRouter
    max_tokens=1024,
    temperature=0.7,
    additional_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Python ChatBot"
    }
)

# Завантажуємо дані для індексу (якщо потрібно)
try:
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(llm=llm)
except Exception as e:
    print(f"Помилка при завантаженні даних: {e}")
    query_engine = None

async def generate_stream_response(prompt: str) -> AsyncGenerator[str, None]:
    """Генерує потокову відповідь від LLM"""
    if query_engine:
        # Для векторного індексу потрібен спеціальний обробник
        response = query_engine.query(prompt)
        for word in str(response).split():
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.05)
    else:
        # Прямий запит до LLM з потоковим виводом
        response = llm.stream_complete(prompt)
        for chunk in response:
            yield f"data: {chunk.delta} \n\n"
            await asyncio.sleep(0.05)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    """Звичайний запит без потокового виводу"""
    try:
        if query_engine:
            response = query_engine.query(message)
        else:
            response = llm.complete(message)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-stream")
async def chat_stream(message: str = Form(...)):
    """Потоковий запит з Server-Sent Events (SSE)"""
    return StreamingResponse(
        generate_stream_response(message),
        media_type="text/event-stream",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)