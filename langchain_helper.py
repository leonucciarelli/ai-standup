from __future__ import annotations
from typing import Any, Dict, List, TypedDict
import os, json

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector import get_retriever
import wikipedia

MODEL_ID = os.getenv("GEN_MODEL", "llama3.1:latest")

class AgentInfo(TypedDict, total=False):
    predicted_comedian: str
    confidence: float
    rationale: List[str]
    bio: str
    sources: List[str]

def _make_llm() -> OllamaLLM:
    return OllamaLLM(model=MODEL_ID, temperature=0.3)

def _compose_joke(request: str) -> str:
    llm = _make_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are a stand-up comedian. Draw inspiration from these jokes:\n{jokes}\nAudience request: {request}\n- Keep it under 80 words.\n- Avoid slurs.\n- Crisp setup → punchline."
    )
    chain = prompt | llm | StrOutputParser()
    retriever = get_retriever()
    candidates = retriever.invoke(request)
    jokes = "\n".join(doc.page_content for doc in candidates[:6])
    return chain.invoke({"jokes": jokes, "request": request}).strip()

def _extract_json(s: str):
    s = s.strip().strip("`")
    if s.startswith("json"): s = s[4:]
    try:
        return json.loads(s)
    except Exception:
        return {}

def _guess_style(joke: str) -> AgentInfo:
    llm = _make_llm()
    prompt = ChatPromptTemplate.from_template(
        "Return ONLY strict JSON: {\"predicted_comedian\":..., \"confidence\":..., \"rationale\":[...]} for this joke:\n{joke}"
    )
    raw = (prompt | llm | StrOutputParser()).invoke({"joke": joke})
    obj = _extract_json(raw)
    return AgentInfo(predicted_comedian=obj.get("predicted_comedian","unknown"), confidence=obj.get("confidence",0.0), rationale=obj.get("rationale",["No rationale"]))

def _enrich(info: AgentInfo) -> AgentInfo:
    name = info.get("predicted_comedian","")
    if not name or name=="unknown":
        info["bio"], info["sources"] = "",""
        return info
    try:
        wikipedia.set_lang("en")
        page = wikipedia.page(name, auto_suggest=True)
        info["bio"] = wikipedia.summary(page.title, sentences=5)
        info["sources"] = [page.url]
    except Exception:
        info["bio"], info["sources"] = "",""
    return info

def tell_joke(request: str) -> Dict[str, Any]:
    joke = _compose_joke(request)
    info = _guess_style(joke)
    info = _enrich(info)
    return {"joke": joke, "info": info}
