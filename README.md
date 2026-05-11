# LangChain RAG (PDF → Chroma → Gemini)

This project indexes a PDF into a local **Chroma** vector database and answers questions using **Gemini** with retrieved context (RAG).

## Project structure

- `rag_system/index.py`: loads `cv.pdf`, splits into chunks, embeds, and stores in `rag_system/chroma_db/`
- `rag_system/query.py`: loads the persisted Chroma DB and answers your question using retrieved chunks
- `rag_system/cv.pdf`: sample PDF to index (replace with your own)

## Setup

### 1) Create `.env`

Add your Google API key (Gemini):

```bash
GOOGLE_API_KEY=your_key_here
```

You can place it in either:
- `rag_system/.env` (recommended if you run from `rag_system/`)
- `.env` (project root)

(`.env` is ignored by git via `.gitignore`.)

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run

### Index the PDF

Run from the `rag_system/` folder so `cv.pdf` resolves correctly:

```bash
cd rag_system
python index.py
```

This will create/update `rag_system/chroma_db/`.

### Ask questions

```bash
cd rag_system
python query.py
```

## Notes / Troubleshooting

- **Deprecation warning (Chroma)**: this repo uses `langchain-chroma` (`from langchain_chroma import Chroma`) to avoid the deprecated `langchain_community.vectorstores` import.
- **PDF not found**: ensure you run scripts from `rag_system/`, or change the loader path in code.
- **Fresh re-index**: delete `rag_system/chroma_db/` and run `index.py` again.

