# **_Reflection_**

The most challenging part of this project was honestly getting all the tools to work together — it started with a bit of dependency hell. Some Python packages (like langchain, sentence-transformers, and huggingface_hub) kept breaking each other, and it took several reinstall attempts to stabilize the environment. Then, when I tried to run Gemini, it turned out that a billing plan was required to actually use the models. Switching to Ollama and running a local LLaMA 3 model made the setup much more stable and independent from API quotas. Later, after enabling billing, Gemini 2.5 Flash also worked correctly — so now both backends (Gemini + Ollama) are fully functional.
Installing ffmpeg for Whisper was another small but essential step — without it, the audio/video transcription simply wouldn’t work.

Once the system ran, it became clear how many moving parts make a RAG pipeline tick.
I learned that transcription caching saves a lot of time, that chunking and retrieval parameters (k, fetch_k, MMR) directly shape answer quality, and that missing concepts in data (like ColPali) can’t be invented by the model — they must exist in the sources. Adding source visibility and a strict guardrail prompt made the workflow transparent and exposed hallucinations when context was missing.

## **Experiment Summary**

I ran several iterations to observe how dataset size, retrieval range, and the choice of model affected performance.

### Experiment 1 – 4 PDFs + 1 audio

With only a few PDFs and one transcript, the chatbot already handled the “RAG Do’s” and hybrid search questions, but it failed on ColPali, which wasn’t in the data. Increasing recall (k = 8, MMR) added coverage but also noise — more fragments retrieved, not always more accuracy.

### Experiment 2 – Full dataset ( 4 PDFs + all 4 video transcriptions)

After adding every video, the answers became much richer and more specific. The model correctly produced seven “production Do’s” for RAG, explained hybrid search with real metrics (e.g. 5.7% → 1.9% failure drop, +10.5% NDCG@3), and even gave a solid technical explanation of ColPali as a vision-language approach avoiding OCR.
None of these metrics were in the PDFs — they came directly from the transcribed lectures, proving how valuable multimodal ingestion is.

Even though the word ColPali never actually appeared in any file, both Gemini and Ollama tried to reconstruct it. Gemini’s response was remarkably accurate, suggesting that it combined the retrieved context with its own multimodal prior knowledge — showing how RAG + LLM reasoning can work together beyond simple retrieval.

### Gemini vs Ollama – Comparison

| Question                | Gemini (Cloud)                                                                 | Ollama (Local)                                             |
| ----------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| **RAG Do’s**            | Enterprise-level checklist with chunking, governance, cost–latency trade-offs. | Correct but less structured; missed governance aspects.    |
| **Standard vs ColPali** | Technically perfect: contrasted OCR vs VLM with patch embeddings.              | Plausible but speculative; mixed up hybrid text retrieval. |
| **Hybrid vs Vector**    | Precise metrics (+10.5 NDCG@3, +37 % BM25 gain).                               | Conceptually correct, no quantitative backing.             |

**Verdict:** Gemini produces more structured, data-driven and technically precise responses.
Ollama is self-contained and offline-safe — great for experimentation and local testing as a confriamtion from presentations.