# DocChat — RAG-based PDF Q&A Assistant

Ask questions about any PDF and get answers grounded in that document, using
Retrieval-Augmented Generation (RAG). 100% free to run — no paid API keys,
no credit card required anywhere in this stack.

## How it works

1. **Load & split**: Your PDF is loaded and broken into ~800-character chunks
   with overlap (so context isn't lost at chunk boundaries).
2. **Embed**: Each chunk is converted into a vector (a list of numbers
   representing its meaning) using a local, free model
   (`sentence-transformers/all-MiniLM-L6-v2`). This runs on your machine —
   no API call, no cost.
3. **Index**: All chunk vectors are stored in a FAISS index — a fast,
   local similarity-search structure (also free, also local).
4. **Retrieve**: When you ask a question, it's embedded the same way, and
   FAISS finds the chunks whose meaning is closest to your question.
5. **Generate**: Those retrieved chunks + your question are sent to an LLM
   (Llama 3, via Groq's free-tier API) which writes an answer using *only*
   that retrieved context — not its general training knowledge. This is
   what keeps answers grounded in your specific document instead of the
   model making things up.

## Setup

### 1. Get a free Groq API key
Go to https://console.groq.com → sign up (no card needed) → create an API key.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
export GROQ_API_KEY="your_key_here"        # Mac/Linux
set GROQ_API_KEY=your_key_here              # Windows cmd
$env:GROQ_API_KEY="your_key_here"           # Windows PowerShell
```

### 4. Run the app
```bash
streamlit run app.py
```

This opens a browser tab. Upload a PDF, click "Process PDF," then ask
questions in the chat box.

## Try it on your own work

A good demo for interviews: upload your own IEEE paper or coursework PDF
and ask it questions about your own content — you already know the right
answers, so you can immediately judge whether the retrieval is accurate.

## What to actually understand before an interview

You should be able to explain, in your own words:
- Why we *embed* text instead of just doing keyword search (embeddings
  capture meaning/similarity, not just exact word matches — so "car" and
  "automobile" can match even with zero shared words)
- Why we chunk the PDF instead of embedding the whole document at once
  (LLMs have a limited context window, and smaller chunks give more
  precise retrieval)
- Why `temperature=0.1` (low temperature = more deterministic, factual
  output — appropriate for Q&A, not creative writing)
- What would break this: very long/scanned PDFs without OCR, ambiguous
  questions, or questions that need info spanning many scattered chunks

## Possible extensions (mention these as "next steps" in an interview)

- Support multiple PDFs at once
- Add conversation memory so follow-up questions work ("what about page 3?")
- Swap FAISS for a more scalable vector DB (Chroma, Pinecone) for larger
  document sets
- Add a simple evaluation step to measure answer accuracy against known
  Q&A pairs
