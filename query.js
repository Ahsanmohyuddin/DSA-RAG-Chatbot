import * as dotenv from 'dotenv';
dotenv.config();
import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from '@google/genai';

const MODEL = process.env.GEMINI_MODEL || 'gemini-3.1-flash-lite';

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: 'gemini-embedding-001',
});
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

console.log('Index:', process.env.PINECONE_INDEX_NAME, '| LLM:', MODEL);

const History = [];

// Koi call 30 second se zyada atke to error de do
const withTimeout = (p, label, ms = 30000) =>
  Promise.race([
    p,
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error(`${label}: ${ms / 1000}s mein jawab nahi aaya`)), ms)
    ),
  ]);

// Follow-up sawal ("explain it") ko poora sawal banao ("explain avl tree")
async function transformQuery(question) {
  if (History.length === 0) return question; // pehle sawal par zaroorat nahi

  History.push({ role: 'user', parts: [{ text: `Follow Up user Question: ${question}` }] });
  try {
    const response = await withTimeout(
      ai.models.generateContent({
        model: MODEL,
        contents: History,
        config: {
          thinkingConfig: { thinkingLevel: 'MINIMAL' },
          systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
Only output the rewritten question and nothing else.`,
        },
      }),
      'Query rewrite'
    );
    return response.text?.trim() || question;
  } catch (e) {
    console.log('Rewrite fail hua, original sawal use kar raha hun:', e.message);
    return question;
  } finally {
    History.pop();
  }
}

async function chatting(question) {
  const t0 = Date.now();

  // 1. follow-up ko standalone sawal banao
  const standalone = await transformQuery(question);
  if (standalone !== question) console.log(`(Samjha gaya sawal: ${standalone})`);
  const t1 = Date.now();

  // 2. vector banao aur Pinecone mein dhoondo
  const queryVector = await withTimeout(embeddings.embedQuery(standalone), 'Embedding');
  const searchResults = await withTimeout(
    pineconeIndex.query({ topK: 5, vector: queryVector, includeMetadata: true }),
    'Pinecone'
  );
  const t2 = Date.now();

  const matches = searchResults.matches ?? [];
  if (!matches.length) {
    console.log('Koi match nahi mila. Pinecone mein records check karein.');
    return;
  }

  const context = matches
    .map((m) => `[Page ${m.metadata?.page}]\n${m.metadata?.text ?? ''}`)
    .join('\n\n---\n\n');
  const pages = [...new Set(matches.map((m) => m.metadata?.page))].sort((a, b) => a - b);

  // 3. jawab banao (asli sawal history mein jata hai)
  History.push({ role: 'user', parts: [{ text: question }] });
  try {
    const response = await withTimeout(
      ai.models.generateContent({
        model: MODEL,
        contents: History,
        config: {
          thinkingConfig: { thinkingLevel: 'MINIMAL' },
          systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
You will be given a context of relevant information and a user question.
Your task is to answer the user's question based ONLY on the provided context.
If the answer is not in the context, you must say "I could not find the answer in the provided document."
Keep your answers clear, concise, and educational.

Context:
${context}`,
        },
      }),
      'LLM jawab'
    );
    History.push({ role: 'model', parts: [{ text: response.text }] });
    console.log('\n' + response.text);
    console.log(`\n(Pages: ${pages.join(', ')})`);
    console.log(
      `Timing: rewrite ${t1 - t0}ms | search ${t2 - t1}ms | jawab ${Date.now() - t2}ms`
    );
  } catch (e) {
    History.pop();
    console.log('LLM error:', e.message);
  }
}

async function main() {
  while (true) {
    const q = readlineSync.question('\nAsk me anything (exit likhein band karne ke liye)--> ');
    if (q.trim().toLowerCase() === 'exit') break;
    await chatting(q);
  }
}

main();