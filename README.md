# AI Stand-Up Joke Generator 🤖🎤

Portfolio project demoing **RAG (Retrieval-Augmented Generation)** with **LangChain**, **Streamlit**, and *Agent-based Wikipedia integration**.

---

## Features
- Joke generation via RAG (Chroma + embeddings).
- LLM-based comedian style guess.
- Wikipedia enrichment for comedian bio & sources.
- Streamlit UI with debug mode.

---

## Quickstart
```bash
git clone <your-repo>
cd ai-standup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Install Ollama and pull models
ollama pull mxbai-embed-large
ollama pull llama3.1:latest

# Run
streamlit run main.py
```
