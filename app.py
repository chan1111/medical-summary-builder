import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import httpx

app = FastAPI(title="Medical Summary Builder")

AI_BUILDER_TOKEN = os.getenv("AI_BUILDER_TOKEN", "")
AI_BASE_URL = "https://space.ai-builders.com/backend/v1"


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not AI_BUILDER_TOKEN:
        raise HTTPException(status_code=500, detail="AI_BUILDER_TOKEN not configured")

    system_prompt = (
        "You are a helpful medical AI assistant for the Medical Summary Builder tool. "
        "Answer questions about medical documentation, clinical summarization, AI in healthcare, "
        "and how the Medical Summary Builder pipeline works. Be concise and professional. "
        "The pipeline uses LLMs to extract structured medical information from PDF case files "
        "and generate formatted Word document summaries."
    )

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{AI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {AI_BUILDER_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-4-fast",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.message},
                ],
                "max_tokens": 400,
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="AI service error")

    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    return {"reply": reply}


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
