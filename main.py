from fastapi import FastAPI
from openai import OpenAI
import os
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Form
import time
from collections import defaultdict
from fastapi import Request, HTTPException
from datetime import date


load_dotenv()

app = FastAPI(title="MaplePath – Immigration")

templates = Jinja2Templates(directory="templates")

RATE_LIMIT = 5  # requests
RATE_WINDOW = 60  # seconds

ip_requests = defaultdict(list)

def check_rate_limit(request: Request):
    ip = request.client.host
    now = time.time()

    # keep only recent requests
    ip_requests[ip] = [
        t for t in ip_requests[ip] if now - t < RATE_WINDOW
    ]

    if len(ip_requests[ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute."
        )

    ip_requests[ip].append(now)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

def search_ircc(query: str):
    params = {
        "engine": "google",
        "q": f"{query} site:canada.ca",
        "api_key": SERPAPI_KEY,
        "num": 5
    }

    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()

    results = []
    for item in data.get("organic_results", []):
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        })

    return results

def is_personal_advice(question: str) -> bool:
    triggers = [
        "my chances",
        "will i get",
        "can i get pr",
        "am i eligible",
        "my profile",
        "based on my",
        "what should i apply",
        "should i apply",
        "my score",
        "my crs"
    ]
    q = question.lower()
    return any(t in q for t in triggers)


@app.get("/")
def health():
    return {"status": "MaplePath running"}


@app.post("/ask")
def ask(request: Request, question: str):
    check_rate_limit(request)

    # Guardrail: personal immigration advice
    if is_personal_advice(question):
        return {
            "answer": (
                "I can’t provide personalized immigration advice or assess individual eligibility.\n\n"
                "What I *can* do is:\n"
                "• Explain immigration programs\n"
                "• Describe eligibility requirements\n"
                "• List required documents\n"
                "• Share official IRCC sources\n\n"
                "For personal advice, please consult a licensed immigration consultant or lawyer."
            ),
            "sources": [
                "https://www.canada.ca/en/immigration-refugees-citizenship.html"
            ]
        }

    
    sources = search_ircc(question)

    if not sources:
        return {
            "answer": "I could not find reliable IRCC sources for this question.",
            "sources": []
        }

    context = ""
    for idx, src in enumerate(sources, start=1):
        context += f"[{idx}] {src['snippet']}\n"

    prompt = f"""
You are MaplePath – a Canada immigration information assistant.

Rules you MUST follow:
- Use only official IRCC or Government of Canada information.
- Do NOT give legal advice.
- Do NOT assess personal eligibility.
- Do NOT predict approval or refusal.
- If a question requires personal details, say you cannot assess individual cases.
- If information is unclear or unavailable, say: "I don’t have enough official information to answer this."
- Always be clear, calm, neutral, factual, non-authorative and cautious.

Always end answers with:
"\n\n⚠️ Disclaimer: This information is for general guidance only. It is not legal advice. Immigration rules change frequently. Always consult official IRCC sources or a licensed professional."

Sources:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
            ]
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": sources
    }

@app.get("/ui")
def ui(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/ui")
def ui_post(request: Request, question: str = Form(...)):
    result = ask(request, question)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "last_updated": date.today().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
