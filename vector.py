import os
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DATA_CSV = os.getenv("JOKES_CSV", "./data/jokes.csv")
DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_langchain_db")
COLLECTION = os.getenv("VECTOR_COLLECTION", "Jokes")
JOKES_LIMIT = int(os.getenv("JOKES_LIMIT", "4000"))

_embeddings = OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "mxbai-embed-large"))
_vector_store = Chroma(collection_name=COLLECTION, persist_directory=DB_DIR, embedding_function=_embeddings)

def seed_if_empty() -> None:
    try:
        if _vector_store._collection.count() > 0:
            return
    except Exception:
        pass
    df = pd.read_csv(DATA_CSV, nrows=JOKES_LIMIT)
    docs, ids = [], []
    for i, row in df.iterrows():
        text = str(row.get("Joke", "")).strip()
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={"row_id": int(i)}))
        ids.append(str(i))
    _vector_store.add_documents(documents=docs, ids=ids)

@dataclass(frozen=True)
class RetrieverConfig:
    k: int = 8
    use_mmr: bool = True
    score_threshold: Optional[float] = 0.2

def get_retriever(cfg: RetrieverConfig = RetrieverConfig()):
    seed_if_empty()
    search_type = "mmr" if cfg.use_mmr else "similarity"
    kwargs = {"k": cfg.k}
    if cfg.score_threshold is not None:
        kwargs["score_threshold"] = cfg.score_threshold
    return _vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)
