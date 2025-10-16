# app.py  — bridge Crisp <-> LLM (senza Webhooks)
import os, base64, logging, requests, pathlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# --- bootstrap ambiente e working dir ---
load_dotenv()
os.chdir(pathlib.Path(__file__).resolve().parent)

# Se non c'è OPENAI_API_KEY ma c'è TOGETHER_API_KEY, mappa in automatico
if not os.getenv("OPENAI_API_KEY") and os.getenv("TOGETHER_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("TOGETHER_API_KEY")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.together.xyz/v1")

from main import compiled_graph  # deve esportare compiled_graph

CRISP_WEBSITE_ID = os.getenv("CRISP_WEBSITE_ID", "").strip()
CRISP_TOKEN_ID   = os.getenv("CRISP_TOKEN_IDENTIFIER", "").strip()
CRISP_TOKEN_KEY  = os.getenv("CRISP_TOKEN_KEY", "").strip()

if not (CRISP_WEBSITE_ID and CRISP_TOKEN_ID and CRISP_TOKEN_KEY):
    raise RuntimeError("Config mancante: CRISP_WEBSITE_ID / CRISP_TOKEN_IDENTIFIER / CRISP_TOKEN_KEY")

BASIC = base64.b64encode(f"{CRISP_TOKEN_ID}:{CRISP_TOKEN_KEY}".encode()).decode()

app = FastAPI(title="CMH LLM Bridge (No Webhooks)")
log = logging.getLogger("uvicorn.error")

# --- FastAPI + CORS ---
raw = os.getenv("ALLOWED_ORIGINS", "")
allow = [o.strip() for o in raw.split(",") if o.strip()]
if not allow:
    # fallback sicuro: domini del sito + localhost e ngrok/fly (da aggiornare dopo il launch)
    allow = [
        "https://www.chinamotorhub.com",
        "https://chinamotorhub.com",
        "http://localhost:8000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

def send_text_to_crisp(session_id: str, text: str):
    """Invia un messaggio operatore nella conversazione Crisp target."""
    url = f"https://api.crisp.chat/v1/website/{CRISP_WEBSITE_ID}/conversation/{session_id}/message"
    headers = {
        "Authorization": f"Basic {BASIC}",
        "X-Crisp-Tier": "plugin",
        "Content-Type": "application/json",
    }
    payload = {"type": "text", "from": "operator", "origin": "chat", "content": text}
    r = requests.post(url, json=payload, headers=headers, timeout=20)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        log.error("Errore invio a Crisp: %s | %s", e, r.text)
        raise

def call_llm(text: str) -> str:
    try:
        out = compiled_graph.invoke({"input": text})
        if isinstance(out, dict) and "output" in out:
            return out["output"]
        return str(out)
    except Exception as e:
        print("LLM ERROR:", e)
        return "Mi dispiace, ho avuto un problema nel generare la risposta."

@app.get("/health")
def health():
    """Endpoint di test per verificare che il backend sia attivo."""
    return {"status": "ok"}

@app.post("/llm_echo")
async def llm_echo(payload: dict):
    """Test rapido della LLM senza inviare nulla a Crisp."""
    text = (payload.get("text") or "").strip()
    reply = call_llm(text) if text else "Nessun testo"
    return {"reply": reply}

# --- NUOVO ENDPOINT USATO DALLO SNIPPET NEL FOOTER ---
@app.post("/llm_ui")
async def llm_ui(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        return {"reply": ""}
    return {"reply": call_llm(text)}

@app.post("/llm")
async def llm_endpoint(req: Request):
    """Riceve {session_id, text} dallo snippet JS e risponde in chat via REST."""
    data = await req.json()
    session_id = (data.get("session_id") or "").strip()
    text       = (data.get("text") or "").strip()
    if not session_id or not text:
        raise HTTPException(status_code=400, detail="session_id o text mancanti")

    # (opzionale) qui potresti inviare "typing..." con endpoint compose start/stop

    reply = call_llm(text)
    try:
        send_text_to_crisp(session_id, reply)
    except Exception as e:
        print(f"[CRISP] warning: {e}")
    return {"ok": True}