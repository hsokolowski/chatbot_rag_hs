import os
import shutil
from pathlib import Path
import argparse

from dotenv import load_dotenv
import whisper

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# -------------------------------------------------------------------
# config
# -------------------------------------------------------------------

DATA_DIR = Path("data")
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# turn off Chroma telemetry noise in console
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

# Prompt: small and opinionated – answer only from context, otherwise say you don't know.
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You answer ONLY from the provided context. If the context is not sufficient, use your general knowledge to give a reasonable answer, but note it clearly.\"\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)

# -------------------------------------------------------------------
# tiny logging helpers
# -------------------------------------------------------------------

def info(msg: str):
    print(f"[INFO] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")

def tip(msg: str):
    print(f"[TIP]  {msg}")

# -------------------------------------------------------------------
# ffmpeg check (needed by Whisper)
# -------------------------------------------------------------------

def ensure_ffmpeg_on_path():
    if shutil.which("ffmpeg"):
        return
    raise FileNotFoundError(
        "ffmpeg not found. Install via `winget install ffmpeg` (Windows) "
        "or see https://ffmpeg.org/download.html"
    )

ensure_ffmpeg_on_path()

# -------------------------------------------------------------------
# transcription
# -------------------------------------------------------------------

def transcribe_audio(file_path: Path) -> str:
    """Transcribe audio/video to text. Cache to <file>.txt next to the source."""
    transcript_path = file_path.with_suffix(file_path.suffix + ".txt")
    if transcript_path.exists():
        info(f"Using cached transcription: {transcript_path.name}")
        return transcript_path.read_text(encoding="utf-8")

    info(f"Transcribing: {file_path.name} (Whisper base)…")
    model = whisper.load_model("base")
    result = model.transcribe(str(file_path))
    text = result["text"].strip()
    transcript_path.write_text(text, encoding="utf-8")
    info(f"Saved transcription → {transcript_path.name}")
    return text

# -------------------------------------------------------------------
# load documents from data/
# -------------------------------------------------------------------

def load_documents() -> list[Document]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {DATA_DIR.resolve()}")

    docs: list[Document] = []
    for f in sorted(DATA_DIR.iterdir()):
        suf = f.suffix.lower()
        if suf == ".pdf":
            info(f"Loading PDF: {f.name}")
            # loader = PyPDFLoader(str(f))
            # pdf_docs = loader.load()
            #
            # # attach minimal metadata for later source display
            # for d in pdf_docs:
            #     d.metadata.setdefault("file", f.name)
            # docs.extend(pdf_docs)

            loader = PyMuPDFLoader(str(f))
            docs.extend(loader.load())

        elif suf in {".mp3", ".wav", ".m4a", ".mp4"}:
            text = transcribe_audio(f)
            docs.append(Document(page_content=text, metadata={"source": "audio", "file": f.name}))

        elif suf in {".txt", ".md"}:
            info(f"Loading text: {f.name}")
            docs.append(Document(page_content=f.read_text(encoding="utf-8"),
                                 metadata={"source": "text", "file": f.name}))
        else:
            # silently ignore other files
            continue

    if not docs:
        warn("No documents found in ./data. Drop PDFs / audio / .txt there and rerun.")
    return docs

# -------------------------------------------------------------------
# vector store build/load
# -------------------------------------------------------------------

def build_vectorstore(rebuild: bool = False) -> Chroma:
    if rebuild and Path(PERSIST_DIR).exists():
        info(f"Removing old vector DB → {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    if Path(PERSIST_DIR).exists():
        info("Loading existing vector DB…")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    info("Building vector DB from data/ …")
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    # Chroma 0.4+ persists automatically; calling persist() is harmless but noisy.
    info("Vector DB ready.")
    return vs

# -------------------------------------------------------------------
# LLM factory (Ollama local or Gemini cloud)
# -------------------------------------------------------------------

def get_llm(name: str = "gemini", api_key: str | None = None):
    name = (name or "gemini").lower()

    if name == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("Missing dependency: install with `pip install -U langchain-ollama`")
        info("Using local model via Ollama: llama3:8b")
        # low temperature → fewer made-up details
        return ChatOllama(model="llama3:8b", temperature=0.1)

    # Gemini (cloud)
    ENV_PATH = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    from langchain_google_genai import ChatGoogleGenerativeAI

    # priority: CLI arg → .env GOOGLE_API_KEY/GEMINI_API_KEY → interactive prompt
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        warn("No GOOGLE_API_KEY / GEMINI_API_KEY found.")
        api_key = input("Paste your Gemini API key: ").strip()
        if not api_key:
            raise RuntimeError("Gemini API key not provided.")
        os.environ["GEMINI_API_KEY"] = api_key
        info("API key loaded from terminal input.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    info(f"Using Gemini model: {model_name}")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,   # keep it conservative for RAG
        api_version="v1",
    )

# -------------------------------------------------------------------
# small helper: answer + sources
# -------------------------------------------------------------------

def print_sources(docs: list[Document], limit: int = 5):
    if not docs:
        print("📚 Sources: (none)\n")
        return
    print("📚 Sources:")
    for i, d in enumerate(docs[:limit], 1):
        meta = d.metadata or {}
        file_ = meta.get("file") or meta.get("source") or "doc"
        page = meta.get("page", "-")
        preview = (d.page_content or "").strip().replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "…"
        print(f"  {i}. {file_} (page {page}) — {preview}")
    print()

# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG chatbot (PDFs + audio).")
    parser.add_argument("--llm", choices=["ollama", "gemini"], default="gemini",
                        help="Choose LLM backend")
    parser.add_argument("--api-key", type=str,
                        help="Gemini API key (optional, overrides .env)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the vector database from data/")
    parser.add_argument("--chat", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--test", action="store_true",
                        help="Run 3 predefined questions and log to file")
    args = parser.parse_args()

    # LLM + vector store
    llm = get_llm(args.llm, api_key=args.api_key)
    vs = build_vectorstore(args.rebuild)

    # retriever: MMR gives diversity; k/fetch_k can be tuned
    #retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    # chain with guardrail prompt + return sources
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    # modes
    if args.chat:
        print("\n💬 Chatbot ready. Type 'exit' to quit.\n")
        while True:
            q = input("> ").strip()
            if q.lower() in {"exit", "quit"}:
                print("👋 Bye!")
                break
            if not q:
                continue

            res = qa.invoke({"query": q})
            answer = res["result"]
            srcs = res.get("source_documents", []) or []

            print(f"\n🧠 {answer}\n")
            print_sources(srcs)

    elif args.test:
        questions = [
            "What are the production 'Do's' for RAG?",
            "What is the difference between standard retrieval and the ColPali approach?",
            "Why is hybrid search better than vector-only search?",
        ]
        info("Running 3 test questions…")
        with open("test_questions.log", "a", encoding="utf-8") as f:
            for q in questions:
                res = qa.invoke({"query": q})
                answer = res["result"]
                srcs = res.get("source_documents", []) or []

                print(f"\nQ: {q}\nA: {answer}\n")
                print_sources(srcs)

                f.write(f"Q: {q}\nA: {answer}\n")
                if srcs:
                    f.write("Sources:\n")
                    for i, d in enumerate(srcs[:5], 1):
                        meta = d.metadata or {}
                        file_ = meta.get("file") or meta.get("source") or "doc"
                        page = meta.get("page", "-")
                        f.write(f"  {i}. {file_} (page {page})\n")
                else:
                    f.write("Sources: (none)\n")
                f.write("\n")
        info("Results saved → test_questions.log")

    else:
        tip("Use one of the flags: --chat or --test")

if __name__ == "__main__":
    main()
