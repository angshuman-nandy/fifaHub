import os
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv()

app = FastAPI()
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
APP_PASSCODE = os.environ.get("APP_PASSCODE", "")
ALLOWED_TOOLS = {"web_search_20250305"}

with open("static/index.html") as f:
    INDEX_HTML = f.read()


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML.replace("__APP_PASSCODE__", APP_PASSCODE))


@app.post("/api/claude")
async def claude_proxy(request: Request):
    if not API_KEY:
        raise HTTPException(500, "Server missing ANTHROPIC_API_KEY")
    body = await request.json()
    # Hard limits so a public Space can't be abused as an open proxy
    if not str(body.get("model", "")).startswith("claude-"):
        raise HTTPException(400, "Invalid model")
    body["max_tokens"] = min(int(body.get("max_tokens", 1000)), 1024)
    body["messages"] = body.get("messages", [])[:4]
    tools = body.get("tools", [])
    if any(t.get("type") not in ALLOWED_TOOLS for t in tools):
        raise HTTPException(400, "Tool not allowed")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            ANTHROPIC_URL,
            json=body,
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
    return JSONResponse(status_code=r.status_code, content=r.json())


@app.get("/healthz")
async def healthz():
    return {"ok": True}
