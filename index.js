import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';

const DIM = 3072;
const EMBED_BATCH = 20;
const UPSERT_BATCH = 50;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const isGood = (v) => Array.isArray(v) && v.length === DIM;

async function embedMany(embeddings, texts) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const vecs = await embeddings.embedDocuments(texts);
      if (vecs.length === texts.length) return vecs;
    } catch (e) {
      console.log(`  error (koshish ${attempt}):`, e.message);
      await sleep(2000 * attempt);
    }
  }
  return null;
}

async function indexdocument() {
  const rawDocs = await new PDFLoader('./dsa.pdf').load();
  console.log('PDF LOADED SUCCESSFULLY, pages:', rawDocs.length);

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allChunks = await textSplitter.splitDocuments(rawDocs);
  // Sirf wo chunks jin mein kam az kam ek harf ya number ho
  const chunkedDocs = allChunks.filter((d) => /[\p{L}\p{N}]/u.test(d.pageContent));
  console.log(`Chunks: ${allChunks.length} total, ${chunkedDocs.length} bache`);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'gemini-embedding-001',
  });

  const goodDocs = [];
  const goodVecs = [];
  let skipped = 0;

  for (let i = 0; i < chunkedDocs.length; i += EMBED_BATCH) {
    const batchDocs = chunkedDocs.slice(i, i + EMBED_BATCH);
    let vecs = await embedMany(embeddings, batchDocs.map((d) => d.pageContent));

    // Batch mein koi bura nikla to ek ek karke try karo
    if (!vecs || !vecs.every(isGood)) {
      vecs = [];
      for (const d of batchDocs) {
        const one = await embedMany(embeddings, [d.pageContent]);
        vecs.push(one ? one[0] : null);
      }
    }

    batchDocs.forEach((d, j) => {
      if (isGood(vecs[j])) {
        goodDocs.push(d);
        goodVecs.push(vecs[j]);
      } else {
        skipped++;
        console.log(
          `SKIP chunk ${i + j} (page ${d.metadata?.loc?.pageNumber}), length ${d.pageContent.length}: ${JSON.stringify(d.pageContent.slice(0, 80))}`
        );
      }
    });
    console.log(`Embedded ${i + batchDocs.length}/${chunkedDocs.length}`);
  }
  console.log(`Embedding khatam: ${goodDocs.length} theek, ${skipped} skip`);

  const records = goodDocs.map((d, i) => ({
    id: `dsa-${i}`,
    values: goodVecs[i],
    metadata: {
      text: d.pageContent,
      page: d.metadata?.loc?.pageNumber ?? 0,
      source: 'dsa.pdf',
    },
  }));

  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
  console.log('Using index:', process.env.PINECONE_INDEX_NAME);

  try {
    await pineconeIndex.deleteAll();
    console.log('Purane records saaf kiye');
  } catch (e) {
    console.log('deleteAll skip:', e.message);
  }

  for (let i = 0; i < records.length; i += UPSERT_BATCH) {
    await pineconeIndex.upsert(records.slice(i, i + UPSERT_BATCH));
    console.log(`Upserted ${Math.min(i + UPSERT_BATCH, records.length)}/${records.length}`);
  }
  console.log('INDEXING MUKAMMAL');
}

indexdocument();