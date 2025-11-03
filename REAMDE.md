# 🧠 RAG Chatbot — PDF + Audio Knowledge Assistant

This project is a small end-to-end **Retrieval-Augmented Generation (RAG)** pipeline.
It loads PDFs and audio files, transcribes them, chunks and embeds the text,
stores the vectors in a local Chroma database, and uses either **Gemini** or **Ollama**
to answer questions directly from your documents.

---

## 📦 What’s inside

| Component              | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| **Whisper**            | Converts audio/video files (mp3, wav, m4a, mp4) into text             |
| **PyPDFLoader**        | Extracts text from PDF files                                          |
| **LangChain + Chroma** | Handles vector embeddings, retrieval, and context search              |
| **Gemini / Ollama**    | LLM backend — choose between local or cloud model                     |
| **RAG prompt**         | Forces the model to use only your data (no guessing or hallucinating) |

---

## ⚙️ Setup

### 1️⃣ Clone & create environment

```bash
git clone https://github.com/yourname/chatbot_rag_hs.git
cd chatbot_rag_hs

python -m venv .venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # macOS/Linux
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install langchain langchain-community langchain-google-genai langchain-ollama
pip install chromadb sentence-transformers whisper-python python-dotenv
```

### 3️⃣ Install ffmpeg

Whisper needs it for audio processing:

* **Windows:** `winget install ffmpeg`
* **macOS:** `brew install ffmpeg`
* **Linux:** `sudo apt install ffmpeg`

---

## 🗂️ Project structure

```
.
├── data/                 # put PDFs, audio or text files here
├── chroma_db/            # generated vector store (auto-created)
├── main.py               # main script
├── requirements.txt
├── .env                  # optional: store GEMINI_API_KEY here
└── test_questions.log    # output from test runs
```

---

## 🚀 Running

### ▶️ Chat interactively

Ask questions based on your data:

```bash
python main.py --chat --llm ollama
```

or with Gemini (you can pass your API key inline):

```bash
python main.py --chat --llm gemini --api-key "YOUR_GEMINI_KEY"
```

### 🧪 Run predefined test questions

This runs 3 fixed RAG-related questions and logs answers to `test_questions.log`:

```bash
python main.py --test --llm ollama
```

### 🔁 Rebuild the vector database

If you add new files to `/data` or want to re-embed everything:

```bash
python main.py --rebuild --llm ollama --test
```

---

## 💬 Prompt behavior

The system uses a guardrail prompt that forces the model to answer only using
context retrieved from your files.

If the answer isn’t in the data, it responds with:

> “I don’t know based on the provided documents.”

This helps reduce hallucinations and ensures factual accuracy.

---

## 🧭 Tips to reduce hallucinations

* Use a **strict prompt** that forbids guessing.
* Lower the **temperature** (0.0–0.2).
* Combine **vector + keyword search (hybrid)** for stronger retrieval quality.
* Evaluate regularly (recall@k, precision).
* Keep embeddings updated when adding new data.

---

## ⚙️ Command line options

| Flag                    | Description                                                                            | Example               |
| ----------------------- | -------------------------------------------------------------------------------------- | --------------------- |
| `--llm {gemini,ollama}` | Choose which model to use. Defaults to `gemini`.                                       | `--llm ollama`        |
| `--api-key <key>`       | Pass your Gemini API key directly via CLI. Overrides `.env`.                           | `--api-key sk-xxxxxx` |
| `--chat`                | Starts interactive chat mode.                                                          | `--chat --llm ollama` |
| `--test`                | Runs the predefined 3 test questions and saves results to a log file.                  | `--test`              |
| `--rebuild`             | Forces a rebuild of the Chroma vector database. Use when you add new files to `/data`. | `--rebuild --test`    |
| `-h` / `--help`         | Displays all available options and exits.                                              | `--help`              |

---

## 🧠 Example usages

#### 1️⃣ Run local Ollama chat

```bash
python main.py --chat --llm ollama
```

#### 2️⃣ Run Gemini chat (inline API key)

```bash
python main.py --chat --llm gemini --api-key "YOUR_GEMINI_API_KEY"
```

#### 3️⃣ Rebuild and run test questions

```bash
python main.py --rebuild --test --llm ollama
```

#### 4️⃣ Just rebuild vector DB

```bash
python main.py --rebuild
```
