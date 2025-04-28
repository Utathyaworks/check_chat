# check_chat

# 🚀 Conversational RAG with Chat History, MLflow Tracking, and REST API

An advanced Conversational RAG (Retrieval-Augmented Generation) system with full **chat history**, **MLflow experiment tracking**, and a **REST API endpoint**.

## ✨ Features

- 💬 **Chat history memory**: Supportsconversation across multiple user turns.
- 📚 **Context-aware retrieval**: Search past interactions to enrich responses.
- 🧠 **FAISS** vector database for efficient document retrieval.
- ⚡ **Groq LLM (Gemma2-9b-It)** for smart response generation.
- 📈 **MLflow integration** for logging experiments, model interactions, and evaluations.
- 🛢️ **SQLite database** for structured storage of interactions (questions, responses, feedback, timestamps).
- 🔥 **REST API** served on `localhost:5005` for external access and integration.
- 📊 Easy viewing of all chat logs and latest conversations.
- 🛠️ Modular, production-grade code.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Streamlit | Frontend web app |
| LangChain | LLM chaining and retrieval |
| FAISS | Vector store for retrieval |
| Groq LLM / OLLAMA / OpenAI| Large Language Model backend |
| MLflow | Experiment and model tracking |
| SQLite3 | Interaction logging |
| Python | Core logic and API backend |

---

## 🗃 Directory Structure

```bash
.
├── app.py                # Main Streamlit app for frontend UI
├── api.py         # REST API backend (Flask/FastAPI) for programmatic access
├── database.py           # SQLite database handling interaction logs
├── test.py       # MLflow utility functions for tracking experiments and models
├── data/                 # Folder to store saved models, logs, and vector stores
├── requirements.txt      # Python dependencies needed for the project
├── README.md             # Project documentation
└── .env                  # Environment variables file (API keys, tokens)
