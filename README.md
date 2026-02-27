# Study Assistant — RAG App

A conversational Q&A app that lets you chat with your own study notes. Built this because I was tired of ctrl+F-ing through 100+ page PDFs the night before exams.

---

## What it does

Upload any PDF (lecture notes, textbook chapters, whatever) and ask it questions in plain english. It pulls the relevant parts from your document and gives you a proper answer instead of just dumping text at you.

Built this as a personal project to actually understand how RAG works — ended up being one of the more useful things I've made.

---

## Stack

- **LangChain** — ties everything together
- **Groq + LLaMA 3.1** — the language model that generates answers
- **ChromaDB** — stores and searches the document embeddings
- **HuggingFace Sentence Transformers** — converts text to embeddings
- **Streamlit** — the chat interface
- **Python 3.12**

---

## How to run it locally

**1. Clone the repo and go into the folder**
```bash
git clone <your-repo-url>
cd ragproj
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install langchain langchain-community langchain-groq langchain-text-splitters langchain-core chromadb pypdf streamlit python-dotenv sentence-transformers
```

**4. Add your API key**

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run app.py
```

Opens in your browser at `localhost:8501`

---

## How to use it

- Drop any PDF into the sidebar upload button
- Wait a few seconds while it processes
- Start asking questions in the chat box
- If no PDF is uploaded it defaults to the included notes.pdf

---

## How it works (the short version)

When you upload a PDF, the app breaks it into small chunks and converts each chunk into a vector (basically a list of numbers that captures the meaning of the text). These get stored in ChromaDB.

When you ask a question, it converts your question into a vector too, finds the chunks that are most similar, and passes those to the LLM along with your question. The LLM then generates an answer based on what it found.

This way it's not just guessing — it's actually reading your notes to answer you.

---

## Project structure

```
ragproj/
├── app.py              # main streamlit app
├── rag_pipeline.py     # pipeline without UI (for testing)
├── load_pdf.py         # PDF loading test script
├── test_gemini.py      # LLM connection test
├── notes.pdf           # default document
├── .env                # API keys (not committed)
└── chroma_db/          # vector store (auto generated)
```

---

## Known limitations

- Works best with text-based PDFs. Scanned/image PDFs have limited text extraction so answers may be incomplete
- Free tier on Groq has rate limits — if you spam questions fast you might hit them
- ChromaDB is stored locally, resets if you delete the folder

---

## Why I built this

I'm in 3rd year CSE and wanted to actually understand what RAG means beyond the buzzword. Building something from scratch that works is the fastest way I know to learn. Also genuinely useful for studying.

---

## License

MIT — do whatever you want with it
