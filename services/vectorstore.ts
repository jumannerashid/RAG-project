// services/vectorstore.ts
import { embedText } from "./embedding.ts";

// Corpus to store document chunks and their embeddings
const corpus: { id: string; text: string; embedding: number[] }[] = [];



/** Split into smaller chunks */
export function chunkText(text: string, chunkSize = 200, overlap = 40): string[] {
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    start += chunkSize - overlap;
  }
  return chunks;
}

/** Retry with exponential backoff */
async function embedWithRetry(
  text: string,
  maxRetries = 6,
  baseDelay = 800
): Promise<number[] | null> {
  let delay = baseDelay;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const vectors = await embedText([text]);
      if (vectors?.[0]?.length > 0) {
        return vectors[0];
      }
    } catch (err) {
      console.warn(`⚠️ Embed attempt ${attempt} failed:`, (err as Error).message);
    }
    console.log(`⏳ Retrying in ${delay}ms...`);
    await sleep(delay);
    delay = Math.min(delay * 1.8, 5000); // cap at 5s
  }
  return null;
}

/** Add document with retries */
export async function addToCorpus(id: string, text: string) {
  if (!text.trim()) return;
  const chunks = chunkText(text);
  console.log(`📦 Splitting ${id} into ${chunks.length} chunk(s)`);

  for (let i = 0; i < chunks.length; i++) {
    const vector = await embedWithRetry(chunks[i]);
    if (!vector) {
      console.error(`❌ Gave up on chunk ${i} of ${id} after retries`);
      continue;
    }
    corpus.push({ id: `${id}_chunk${i}`, text: chunks[i], embedding: vector });
    console.log(`✅ Embedded chunk ${i} of ${id} (dim: ${vector.length})`);
    await sleep(400); // throttle between chunks
  }
}

/** Cosine similarity */
function cosineSim(a: number[], b: number[]) {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return dot / (magA * magB + 1e-10);
}

/** Query with retry */
export async function queryCorpus(query: string, topK = 3) {
  if (corpus.length === 0) return [];
  const qVec = await embedWithRetry(query);
  if (!qVec) return [];
  return corpus
    .map(item => ({ ...item, score: cosineSim(qVec, item.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

export function getCorpus() {
  return corpus;
}

export function getCorpusSize() {
  return corpus.length;
}

function sleep(delay: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, delay));
}
