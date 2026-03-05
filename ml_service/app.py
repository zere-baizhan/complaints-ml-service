from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from typing import Optional, Dict, Any

category_model = joblib.load("category_model.joblib")

CRITICAL_TRIGGERS = [
    r"\bвзрыв\b", r"\bгорит\b", r"\bпожар\b", r"\bдым\b", r"\bугар\b",
    r"\bискрит\b", r"\bкоротк(ое|ий)\b"
]
URGENT_TRIGGERS = [
    r"\bпахнет газом\b", r"\bзапах газа\b", r"\bутечк(а|и)\s*газ(а|ом)\b",
    r"\bтечет вода\b", r"\bс потолка теч(ет|ь)\b", r"\bпротечк(а|и)\b",
    r"\bсильно теч(ет|ь)\b", r"\bпрорв(ало|ало)\b", r"\bзатопил(о|а)\b",
    r"\bлифт\b.*\bзастр(ял|яли)\b",
    r"\bнет свет(а|)\b", r"\bвырубил(о|ся)\b"
]

def _has_any(patterns, text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)

PUSH_TEMPLATES = {
    ("газ", "urgent"): "Похоже на утечку газа. Откройте окна, НЕ включайте/не выключайте электроприборы, покиньте квартиру и позвоните 104/112. Специалисты уведомлены.",
    ("газ", "critical"): "КРИТИЧНО: возможна опасность из-за газа. Немедленно покиньте помещение, не включайте свет/технику, предупредите соседей и позвоните 104/112. Бригада выехала.",
    ("вода", "urgent"): "Похоже на сильную протечку. По возможности перекройте воду и уберите электроприборы из зоны воды. Заявка передана специалистам.",
    ("вода", "normal"): "Заявка принята. Специалисты рассмотрят обращение и свяжутся при необходимости.",
    ("свет", "urgent"): "Если есть запах гари/дым — немедленно отключите питание (если безопасно) и позвоните 112. Заявка передана специалистам.",
    ("свет", "normal"): "Заявка по электроснабжению принята. Специалисты проверят и сообщат статус.",
    ("лифт", "urgent"): "Если кто-то застрял в лифте — не пытайтесь открыть двери самостоятельно. Сообщите диспетчеру/112. Заявка передана специалистам.",
    ("лифт", "normal"): "Заявка по лифту принята. Специалисты проверят оборудование."
}

def decide_urgency(text: str, category: str, spam_count_last_10min: int = 0) -> str:
    if spam_count_last_10min >= 3:
        return "critical"
    if _has_any(CRITICAL_TRIGGERS, text):
        return "critical"
    if category == "газ" and _has_any(URGENT_TRIGGERS, text):
        return "urgent"
    if _has_any(URGENT_TRIGGERS, text):
        return "urgent"
    return "normal"

def get_push(category: str, urgency: str) -> str:
    return PUSH_TEMPLATES.get((category, urgency), "Заявка принята. Специалисты рассмотрят обращение и при необходимости свяжутся с вами.")

app = FastAPI(title="Complaints ML Service", version="1.0")

class ComplaintIn(BaseModel):
    title: str
    description: str
    spam_count_last_10min: int = 0
    user_id: Optional[str] = None
    address: Optional[dict] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify")
def classify(inp: ComplaintIn) -> Dict[str, Any]:
    text = (inp.title.strip() + " " + inp.description.strip()).strip()
    category = category_model.predict([text])[0]

    conf = None
    if hasattr(category_model.named_steps["clf"], "predict_proba"):
        proba = category_model.predict_proba([text])[0]
        classes = category_model.named_steps["clf"].classes_
        conf = float(proba[list(classes).index(category)])

    urgency = decide_urgency(text, category, inp.spam_count_last_10min)
    push = get_push(category, urgency)

    return {
        "category": category,
        "confidence": conf,
        "urgency": urgency,
        "push_message": push
    }