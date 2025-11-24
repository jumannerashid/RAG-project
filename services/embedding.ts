// services/embedding.ts
//import "https://deno.land/x/dotenv/load.ts";

const HF_API_KEY = Deno.env.get("HF_API_KEY");
if (!HF_API_KEY) throw new Error("HF_API_KEY is not set in .env file");

// Model & endpoint — FIXED PATH: /models/${MODEL}/pipeline/feature-extraction
const MODEL = "sentence-transformers/all-MiniLM-L6-v2";
const PIPELINE_ENDPOINT = `https://router.huggingface.co/hf-inference/models/${MODEL}/pipeline/feature-extraction`;

// config
const MAX_TOKEN_LENGTH = 512; // approx token limit
const EXPECTED_DIMENSION = 384;
const BATCH_SIZE = 8;
const MAX_ATTEMPTS = 5;

function _estimateTokens(text: string): number {
  return Math.ceil(text.length / 3);
}
function truncateToTokenLimit(text: string, maxTokens = MAX_TOKEN_LENGTH): string {
  const maxChars = maxTokens * 3;
  if (text.length <= maxChars) return text;
  console.warn(`Truncating text to ${maxChars} chars (~${maxTokens} tokens): ${text.slice(0, 50)}...`);
  return text.slice(0, maxChars).trim();
}

export function cosineSim(a: number[], b: number[]): number {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return magA && magB ? dot / (magA * magB) : 0;
}

/**
 * Try to normalize various HF responses into an array of number[] embeddings.
 * Accepts:
 *  - [ [numbers], [numbers], ... ]
 *  - [ {embedding: [numbers]}, ... ]
 *  - nested forms: [ [[numbers]] , ... ]  — common for sentence-transformers
 */
function normalizeEmbeddings(resp: any, requestedBatchLen: number): number[][] {
  if (!resp) return new Array(requestedBatchLen).fill(null).map(() => new Array(EXPECTED_DIMENSION).fill(0));

  // If the response is an object with "embedding" fields
  if (Array.isArray(resp) && resp.length && typeof resp[0] === "object" && Array.isArray(resp[0].embedding)) {
    return resp.map((r: any) => r.embedding.slice(0, EXPECTED_DIMENSION));
  }

  // If response is array of arrays (possibly nested) — ENHANCED FOR HF NESTING
  if (Array.isArray(resp) && Array.isArray(resp[0])) {
    // Flatten deep nesting like [[[vec]]] -> [vec]
    let maybeFlattened = resp;
    while (Array.isArray(maybeFlattened[0]) && Array.isArray(maybeFlattened[0][0])) {
      maybeFlattened = maybeFlattened.map((r: any) => r[0]);
    }
    // Now flatten single-nest like [[vec]] -> [vec]
    maybeFlattened = maybeFlattened.map((r: any) => Array.isArray(r[0]) ? r[0] : r);
    const normalized = maybeFlattened.map((v: any) =>
      Array.isArray(v) ? (v.length === EXPECTED_DIMENSION ? v : padOrTrim(v)) : new Array(EXPECTED_DIMENSION).fill(0)
    );
    // ADDED: Quick dim validation
    const actualDim = normalized[0]?.length;
    if (actualDim && actualDim !== EXPECTED_DIMENSION) {
      console.warn(`Embedding dim mismatch: Got ${actualDim}, expected ${EXPECTED_DIMENSION}. Truncating/padding.`);
    }
    return normalized;
  }

  // If the response is an object with "data" or similar
  if (resp.data && Array.isArray(resp.data) && resp.data[0].embedding) {
    return resp.data.map((d: any) => d.embedding.slice(0, EXPECTED_DIMENSION));
  }

  // Unexpected: return zero vectors
  return new Array(requestedBatchLen).fill(null).map(() => new Array(EXPECTED_DIMENSION).fill(0));
}

function padOrTrim(arr: number[]) {
  if (!Array.isArray(arr)) return new Array(EXPECTED_DIMENSION).fill(0);
  if (arr.length === EXPECTED_DIMENSION) return arr;
  if (arr.length > EXPECTED_DIMENSION) return arr.slice(0, EXPECTED_DIMENSION);
  return arr.concat(new Array(EXPECTED_DIMENSION - arr.length).fill(0));
}

/** SIMPLIFIED: Single robust HF call to feature-extraction pipeline. No more multi-payload loop. */
async function tryGetEmbeddingsFromHF(batch: string[]): Promise<number[][]> {
  const body = {
    inputs: batch,  // Array of strings — standard for feature-extraction
    options: { wait_for_model: true }  // Wait for cold start
    // Optional: Add normalize: true for L2-norm (cosine-ready)
  };

  console.log(`Attempting HF call -> feature-extraction pipeline (${PIPELINE_ENDPOINT}) len=${batch.length}`);
  const res = await fetch(PIPELINE_ENDPOINT, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    console.warn(`HF API attempt returned ${res.status}: ${text}`);
    // Common: 429 (rate limit) or 503 (cold start) — let outer retry handle
    throw new Error(`HF API ${res.status}: ${text}`);
  }

  const json = await res.json();
  const normalized = normalizeEmbeddings(json, batch.length);
  if (Array.isArray(normalized) && normalized.length === batch.length && normalized[0]?.length === EXPECTED_DIMENSION) {
    console.log(`Success: Got ${batch.length} embeddings of dim ${EXPECTED_DIMENSION}`);
    return normalized;
  } else {
    console.warn(`Unexpected shape: ${JSON.stringify({ len: normalized.length, dim: normalized[0]?.length })} — returning zeros`);
    return new Array(batch.length).fill(null).map(() => new Array(EXPECTED_DIMENSION).fill(0));
  }
}

/**
 * Public function: embedChunks — UNCHANGED LOGIC, but now uses simplified tryGetEmbeddingsFromHF
 */
export async function embedChunks(chunks: string[]): Promise<number[][]> {
  if (!chunks || chunks.length === 0) return [];

  // Preprocess chunks: truncate, filter tiny items
  const validChunkData = chunks.map((chunk, idx) => {
    const t = truncateToTokenLimit((chunk ?? "").trim());
    const ok = typeof t === "string" && t.length >= 10 && !/^\{width=/.test(t);
    if (!ok) console.warn(`Skipping chunk ${idx}: too short/invalid.`);
    return { originalIndex: idx, text: t, isValid: ok };
  });

  const validTexts = validChunkData.filter(c => c.isValid).map(c => c.text);
  if (validTexts.length === 0) {
    console.warn("No valid chunks — returning zero vectors for all inputs.");
    return chunks.map(() => new Array(EXPECTED_DIMENSION).fill(0));
  }

  const allEmbeddingsForValid: number[][] = [];
  for (let start = 0; start < validTexts.length; start += BATCH_SIZE) {
    const batch = validTexts.slice(start, start + BATCH_SIZE);
    console.log(`Processing batch ${Math.floor(start / BATCH_SIZE) + 1} (len=${batch.length})`);

    let lastErr: unknown = null;
    let attempt = 0;
    while (attempt < MAX_ATTEMPTS) {
      try {
        // Now a single, robust call
        const batchEmb = await tryGetEmbeddingsFromHF(batch);
        // ensure dims
        const safe = batchEmb.map(e => (Array.isArray(e) ? padOrTrim(e) : new Array(EXPECTED_DIMENSION).fill(0)));
        allEmbeddingsForValid.push(...safe);
        break;
      } catch (err) {
        attempt++;
        lastErr = err;
        const wait = Math.pow(2, attempt) * 200;
        console.warn(`Attempt ${attempt}/${MAX_ATTEMPTS} failed for batch ${Math.floor(start / BATCH_SIZE) + 1}: ${err}. retrying in ${wait}ms`);
        await new Promise(r => setTimeout(r, wait));
      }
    }

    if (attempt === MAX_ATTEMPTS) {
      throw new Error(`Failed to get embeddings for batch ${Math.floor(start / BATCH_SIZE) + 1} after ${MAX_ATTEMPTS} attempts. Last error: ${lastErr}`);
    }
  }

  // Map back to original order. For skipped invalid items, produce zeros.
  return chunks.map((_ch, idx) => {
    const match = validChunkData.find(v => v.originalIndex === idx && v.isValid);
    if (!match) return new Array(EXPECTED_DIMENSION).fill(0);
    // find index among validTexts
    const idxInValid = validTexts.indexOf(match.text);
    return allEmbeddingsForValid[idxInValid] ?? new Array(EXPECTED_DIMENSION).fill(0);
  });
}