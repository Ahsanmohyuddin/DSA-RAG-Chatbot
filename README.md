# DSA RAG Chatbot

Chat with a Data Structures & Algorithms PDF. You ask a question, the app finds the most relevant parts of the PDF, and Google Gemini answers **using only those parts**. If the answer is not in the document, it says so instead of guessing.

Built with **LangChain**, **Pinecone** and the **Google Gemini API**.

## How it works

**Indexing (run once per document)**

1. `PDFLoader` (LangChain) reads the PDF page by page.
2. `RecursiveCharacterTextSplitter` (LangChain) cuts it into chunks of 1000 characters with 200 characters of overlap.
3. Each chunk is turned into a 3072-dimension vector by Gemini (`gemini-embedding-001`).
4. The vectors, plus the chunk text and page number as metadata, are stored in Pinecone.

**Chatting (every question)**

1. If the question is a follow-up ("explain it"), Gemini first rewrites it into a standalone question using the chat history ("Explain what an AVL tree is").
2. The question is embedded with the same Gemini embedding model.
3. Pinecone returns the 5 most similar chunks.
4. Gemini answers using only those chunks as context, and the app prints which pages were used.

## Tech stack

| Part | Tool | Used for |
| --- | --- | --- |
| Loading and splitting | LangChain (`@langchain/community`, `@langchain/textsplitters`) | Read the PDF and split it into chunks |
| Embeddings | Google Gemini `gemini-embedding-001` through `@langchain/google-genai` | Turn text into 3072-dimension vectors |
| Vector database | Pinecone (`@pinecone-database/pinecone`) | Store vectors and search by similarity |
| LLM | Google Gemini through `@google/genai` | Rewrite follow-up questions and write answers |
| Runtime | Node.js (ES modules), `dotenv`, `readline-sync` | Config and the terminal chat loop |

## Project structure

```
index.js          Reads dsa.pdf, creates embeddings and uploads them to Pinecone
query.js          Terminal chat: rewrite question -> search Pinecone -> answer with Gemini
package.json      Dependencies (npm install reads this)
```

## Setup

1. **Clone and install**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   npm install
   ```

   If npm stops with an `ERESOLVE` peer-dependency error, run `npm install --legacy-peer-deps` instead (see problem 7 below).

2. **Add your keys.** Create a file named `.env` in the project folder (it is git-ignored, never commit it) with these lines:

   ```
   GEMINI_API_KEY=your_google_ai_studio_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your-index-name
   GEMINI_MODEL=gemini-3.1-flash-lite
   ```

   What they mean:

   | Variable | Meaning |
   | --- | --- |
   | `GEMINI_API_KEY` | Key from Google AI Studio |
   | `PINECONE_API_KEY` | Key from the Pinecone console |
   | `PINECONE_INDEX_NAME` | Name of the index you create in the next step |
   | `GEMINI_MODEL` | Optional. LLM used for answers |

3. **Create the Pinecone index.** It must be a classic dense vector index with dimension **3072** and metric **cosine** (see problem 1 for why it must be created from code, not from the new console):

   ```js
   import * as dotenv from 'dotenv';
   dotenv.config();
   import { Pinecone } from '@pinecone-database/pinecone';

   const pc = new Pinecone();
   await pc.createIndex({
     name: process.env.PINECONE_INDEX_NAME,
     dimension: 3072,
     metric: 'cosine',
     spec: { serverless: { cloud: 'aws', region: 'us-east-1' } }, // use a cloud/region your plan supports
     waitUntilReady: true,
   });
   ```

   Save it as a file such as `create-index.js` and run it once with `node create-index.js`. You do not need to keep or commit this file after the index exists.

4. **Add your PDF** as `dsa.pdf` in the project folder (PDFs are git-ignored).

5. **Index it and chat**

   ```bash
   node index.js    # one time. Note: it first deletes everything already in the index
   node query.js    # start asking questions, type "exit" to quit
   ```

## Problems we faced and how we fixed them

This project took a lot of debugging because several services changed in 2026. These are the real problems, in the order we hit them.

### 1. Pinecone: "This index has a document schema, so writes must go through the documents API"

- **Symptom:** `PineconeStore.fromDocuments(...)` failed with HTTP 400 when uploading.
- **Cause:** Indexes created in the new Pinecone console are "document-schema" indexes. LangChain's `PineconeStore` writes through the older vectors API, which those indexes reject. This is a known open issue: [langchain-pinecone #158](https://github.com/langchain-ai/langchain-pinecone/issues/158).
- **Fix:** Create a classic vector index from code with `createIndex({ dimension, metric })`. Pinecone says this still creates a legacy vector index that uses the `/vectors/*` API, and that worked.

### 2. "Vector dimension 0 does not match the dimension of the index"

- **Symptom:** Pinecone rejected the upload because some vectors were empty.
- **Cause 1:** The embedding model `text-embedding-004` was shut down on the Gemini API on 14 January 2026. Our best explanation is that LangChain then returned empty vectors instead of showing the 404 (we did not confirm this).
- **Fix 1:** Switch to `gemini-embedding-001`. It returns **3072** dimensions by default (not 768), so the index had to be recreated with dimension 3072.
- **Cause 2:** Even with the right model, at least one chunk kept returning an empty embedding on every retry. The exact content was not confirmed; chunks with no real text (dots, symbols) are the likely reason.
- **Fix 2:** We stopped using `PineconeStore.fromDocuments` and wrote our own indexing loop: embed in small batches, check that every vector has 3072 numbers, retry, fall back to one chunk at a time, skip (and print) any chunk that still fails, then upload in batches of 50 with fixed IDs (`dsa-0`, `dsa-1`, ...) so re-running does not create duplicates.

### 3. Wrong index name in `.env`

- **Symptom:** 404 errors and "API key was rejected" errors that pointed at an index we had already deleted.
- **Cause:** `.env` still had the old index name, so the script kept talking to the old index.
- **Fix and lesson:** Print `process.env.PINECONE_INDEX_NAME` when the script starts. Also remember that a variable already set in Windows overrides `.env`, because `dotenv` does not overwrite existing variables by default.

### 4. The LLM model was retired

- **Symptom:** The answer step failed because `gemini-2.0-flash` no longer exists.
- **Cause:** Google shut down the Gemini 2.0 Flash models on 1 June 2026, and model names keep rotating.
- **Fix:** The model name now comes from the `GEMINI_MODEL` environment variable, so changing it never needs a code edit.

### 5. Answers were very slow

- **Cause:** Newer Gemini 3.x Flash models think before answering, and every question makes three network calls (embedding, Pinecone, LLM).
- **Mitigation:** Use a lighter model, set `thinkingLevel` to `MINIMAL`, retrieve 5 chunks instead of 10, and add a 30-second timeout to every call so a stuck call shows an error instead of hanging silently. The app prints how long each step takes.

### 6. Follow-up questions did not work

- **Symptom:** "What is an AVL tree?" worked, but "explain it" answered "I could not find the answer in the provided document."
- **Cause:** The words "explain it" have no topic, so Pinecone found no relevant chunks.
- **Fix:** Query rewriting. Before searching, Gemini turns the follow-up into a standalone question using the chat history. It is skipped for the first question and falls back to the original question if rewriting fails.

### 7. npm "ERESOLVE" peer-dependency conflict

- **Symptom:** `npm i` failed because `dotenv@18` clashed with an optional peer dependency (`@browserbasehq/stagehand`) of `@langchain/community`, which wants `dotenv@^16`.
- **Fix:** `npm install --legacy-peer-deps`. This project only uses the PDF loader, not the browser-based loaders that need that package.

### 8. Code-structure bugs

- `queryVector` was used outside the function that created it (`ReferenceError`).
- The Gemini answer code was placed after an endless `while (true)` loop, so it never ran, and it used variables (`question`, `context`) that did not exist there.
- **Fix:** Everything for one question (rewrite, embed, search, answer) now lives inside a single `chatting()` function, and the chat history lives outside it.

## Limitations

- Answers come only from the PDF you indexed.
- Scanned PDFs without a text layer will not work (they need OCR first).
- If you change the embedding model, recreate the index with the new dimension and run `index.js` again.
- The LLM and embedding model names will change again over time. Check the Gemini models page when something stops working.

## Ideas for next steps

- Stream the answer word by word.
- Show the source chunk text next to each answer.
- Add reranking or hybrid (keyword + vector) search.
- Build a small web UI instead of the terminal.

## References

- [langchain-pinecone issue #158: document-schema indexes](https://github.com/langchain-ai/langchain-pinecone/issues/158)
- [Pinecone: Move to the Documents API](https://docs.pinecone.io/guides/indexes/move-to-the-documents-api)
- [Gemini API deprecations](https://ai.google.dev/gemini-api/docs/deprecations)
- [Gemini API changelog](https://ai.google.dev/gemini-api/docs/changelog)
