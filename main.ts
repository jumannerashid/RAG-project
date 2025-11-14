import "https://deno.land/x/dotenv/load.ts";
import { Hono } from "https://deno.land/x/hono@v3.1.5/mod.ts";
import { z } from "https://deno.land/x/zod/mod.ts";

import { embedChunks } from "./services/embedding.ts";
import { upsertToPinecone, queryPinecone } from "./services/pinecone.ts";
import { chatWithHF } from "./services/llm.ts";

const app = new Hono();
const ChatSchema = z.object({ query: z.string().min(1) });

// UPDATED: Match the embedding model dimension (sentence-transformers/all-MiniLM-L6-v2 uses 384, not 768)
const NAMESPACE = "my-docs";
const EXPECTED_DIMENSION = 384;

// Type definition for Pinecone query response
interface PineconeMatch {
  id: string;
  score: number;
  metadata: { text: string; sourceFile: string };
}
interface PineconeResponse {
  matches: PineconeMatch[];
  namespace?: string;
}

// Enable CORS
app.use("*", async (c, next) => {
  c.header("Access-Control-Allow-Origin", "*");
  c.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  c.header("Access-Control-Allow-Headers", "Content-Type");
  if (c.req.method === "OPTIONS") return c.text("", 204);
  await next();
});

// Clean text based on file type
function cleanText(text: string, isMarkdown: boolean): string {
  text = text
    .replace(/!\[.*?\]\(.*?\)/g, "")
    .replace(/\{width="[^"]+" height="[^"]+"\}\s*/g, "")
    .replace(/\+[-+=]+\+/g, "")
    .replace(/\|[-|]+\|/g, "")
    .replace(/>\s+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!isMarkdown) {
    return text
      .replace(/\r\n/g, "\n")
      .replace(/\n{2,}/g, "\n")
      .trim();
  }
  return text
    .replace(/---[\s\S]*?---\n*/g, "")
    .replace(/```[\s\S]*?```/g, (match) => ` [CODE_BLOCK: ${match.replace(/```/g, "").trim()}] `)
    .replace(/^#{1,6}\s*(.*)$/gm, "$1")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1")
    .replace(/^\s*[-+*]\s+/gm, "- ")
    .replace(/\n{2,}/g, "\n")
    .replace(/\|([^\|]*)\|/g, "$1")
    .replace(/^\s*>+\s*/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

// Basic token estimation: ~3 chars per token
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3);
}

// Chunk text for better embeddings
function chunkText(text: string, maxLength: number = 500): string[] {
  const chunks: string[] = [];
  let currentChunk = "";
  const sentences = text.split(/(?<=\.|\!|\?)\s+/);
  for (const sentence of sentences) {
    if ((currentChunk + sentence).length > maxLength) {
      if (currentChunk.trim()) {
        const trimmed = currentChunk.trim().slice(0, maxLength);
        const estTokens = estimateTokens(trimmed);
        chunks.push(trimmed);
        console.log(`Chunk created, size: ${trimmed.length} chars, ~${estTokens} tokens, content: ${trimmed.slice(0, 50)}...`);
      }
      currentChunk = sentence;
      if (sentence.length > maxLength) {
        console.warn(`Sentence too long (${sentence.length} chars, ~${estimateTokens(sentence)} tokens): ${sentence.slice(0, 50)}...`);
        const subChunks = [];
        let subChunk = "";
        const words = sentence.split(/\s+/);
        for (const word of words) {
          if ((subChunk + word).length > maxLength) {
            if (subChunk.trim()) {
              const trimmedSub = subChunk.trim().slice(0, maxLength);
              const estTokens = estimateTokens(trimmedSub);
              subChunks.push(trimmedSub);
              console.log(`Sub-chunk created, size: ${trimmedSub.length} chars, ~${estTokens} tokens, content: ${trimmedSub.slice(0, 50)}...`);
            }
            subChunk = word;
          } else {
            subChunk += " " + word;
          }
        }
        if (subChunk.trim()) {
          const trimmedSub = subChunk.trim().slice(0, maxLength);
          const estTokens = estimateTokens(trimmedSub);
          subChunks.push(trimmedSub);
          console.log(`Sub-chunk created, size: ${trimmedSub.length} chars, ~${estTokens} tokens, content: ${trimmedSub.slice(0, 50)}...`);
        }
        chunks.push(...subChunks);
      }
    } else {
      currentChunk += " " + sentence;
    }
  }
  if (currentChunk.trim()) {
    const trimmed = currentChunk.trim().slice(0, maxLength);
    const estTokens = estimateTokens(trimmed);
    chunks.push(trimmed);
    console.log(`Final chunk created, size: ${trimmed.length} chars, ~${estTokens} tokens, content: ${trimmed.slice(0, 50)}...`);
  }
  return chunks.filter(chunk => chunk.length >= 10 && !/^\{width=/.test(chunk));
}

// Normalize embeddings for better similarity matching
function normalizeVector(vector: number[]): number[] {
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  return magnitude === 0 ? vector : vector.map(val => val / magnitude);
}

// Load and process files from ./data
async function _loadCorpus() {
  const dir = "./data";
  const files: { id: string; text: string; sourceFile: string }[] = [];
  try {
    console.log("Scanning directory:", dir);
    let fileCount = 0;
    let mdCount = 0;
    let txtCount = 0;

    for await (const entry of Deno.readDir(dir)) {
      if (entry.isFile && (entry.name.endsWith(".txt") || entry.name.endsWith(".md"))) {
        fileCount++;
        const isMarkdown = entry.name.endsWith(".md");
        if (isMarkdown) mdCount++;
        if (entry.name.endsWith(".txt")) txtCount++;
        
        try {
          const rawText = await Deno.readTextFile(`${dir}/${entry.name}`);
          if (!rawText.trim()) {
            console.warn(`Skipping ${entry.name}: empty file`);
            continue;
          }
          console.log(`Reading ${entry.name} (${rawText.length} chars)`);
          const cleanedText = cleanText(rawText, isMarkdown);
          if (cleanedText.length < 10) {
            console.warn(`Skipping ${entry.name}: too short after cleaning (${cleanedText.length} chars)`);
            continue;
          }
          console.log(`Processing ${entry.name}: ${cleanedText.slice(0, 100)}...`);
          console.log(`Full cleaned text for ${entry.name}: ${cleanedText.slice(0, 500)}...`);
          const chunks = chunkText(cleanedText);
          console.log(`Created ${chunks.length} chunks for ${entry.name}`);
          chunks.forEach((chunk, index) => {
            const id = `${entry.name}-${index}`;
            files.push({ id, text: chunk, sourceFile: entry.name });
          });
        } catch (err: unknown) {
          console.error(`Error reading ${entry.name}:`, err);
        }
      }
    }

    console.log(`Found ${fileCount} files (${mdCount} .md, ${txtCount} .txt)`);
    if (files.length === 0) {
      console.log("No valid documents found in ./data");
      return;
    }

    // Embed all texts
    console.log("Embedding documents:", files.map(f => f.id));
    const chunks = files.map(f => f.text);
    const rawEmbeddings = await embedChunks(chunks) as number[][];
    if (!rawEmbeddings || rawEmbeddings.length !== chunks.length) {
      throw new Error("Embedding failed: Invalid response from HuggingFace");
    }
    // UPDATED: Log actual dimension from embeddings (should be 384)
    const actualDim = rawEmbeddings[0]?.length || 0;
    console.log(`Embedding dimensions: ${actualDim} (expected: ${EXPECTED_DIMENSION})`);
    if (actualDim !== EXPECTED_DIMENSION) {
      console.warn(`Dimension mismatch: Got ${actualDim}, expected ${EXPECTED_DIMENSION}. Proceeding but check model config.`);
    }

    // Filter out invalid embeddings
    const embeddings = rawEmbeddings
      .map((vector, index) => {
        if (!vector || vector.length !== EXPECTED_DIMENSION || vector.every(v => v === 0)) {
          console.warn(`Skipping invalid or zero-filled embedding for ${files[index].id}: vector length ${vector?.length || 0}`);
          return null;
        }
        return {
          id: files[index].id,
          vector: normalizeVector(vector),
          metadata: { text: files[index].text, sourceFile: files[index].sourceFile },
        };
      })
      .filter((e): e is NonNullable<typeof e> => e !== null);

    if (embeddings.length === 0) {
      throw new Error("No valid embeddings generated");
    }

    // Upsert to Pinecone
    for (const embedding of embeddings) {
      console.log("Upserting to Pinecone:", embedding.id, "with sourceFile:", embedding.metadata.sourceFile);
      await upsertToPinecone({ id: embedding.id, vector: embedding, namespace: NAMESPACE, upsertMetadata: true });
    }

    console.log(`✅ Loaded ${embeddings.length} document chunks into Pinecone.`);
  } catch (err: unknown) {
    console.error("Failed to load corpus:", err);
    throw err;
  }
}

// Chat endpoint
app.post("/chat", async (c) => {
  try {
    const body = await c.req.json();
    const parsed = ChatSchema.safeParse(body);
    if (!parsed.success) return c.json({ error: "Invalid input", details: parsed.error }, 400);

    const { query } = parsed.data;

    // Embed query
    console.log("Embedding query:", query);
    const queryEmbeddings = await embedChunks([query]);
    if (!queryEmbeddings || queryEmbeddings.length === 0 || !queryEmbeddings[0] || queryEmbeddings[0].length !== EXPECTED_DIMENSION) {
      const actualQueryDim = queryEmbeddings[0]?.length || 0;
      console.error(`Query embedding invalid: length ${actualQueryDim}, expected ${EXPECTED_DIMENSION}`);
      return c.json({ error: "Failed to embed query" }, 500);
    }
    const queryEmbedding = normalizeVector(queryEmbeddings[0]);

    // Query Pinecone
    console.log("Querying Pinecone for:", query);
    const pineconeResponse = await queryPinecone({
      vector: queryEmbedding,
      topK: 100,  // Increased for broader context if needed
      namespace: NAMESPACE,
    }) as PineconeResponse | PineconeMatch[];

    // Handle Pinecone response
    let results: PineconeMatch[] = [];
    if (Array.isArray(pineconeResponse)) {
      results = pineconeResponse;
    } else if (pineconeResponse && Array.isArray(pineconeResponse.matches)) {
      results = pineconeResponse.matches;
    } else {
      console.warn("Invalid Pinecone response:", pineconeResponse);
      return c.json({ error: "No relevant context found", details: "Invalid Pinecone response" }, 404);
    }

    // Log all results before filtering
    console.log("All Pinecone results before filtering:", results.map(r => ({ id: r.id, score: r.score, sourceFile: r.metadata?.sourceFile || 'unknown' })));
    // UPDATED: Lower threshold to 0.0 (no filter) for testing; raise to 0.6+ later for quality
    results = results.filter((r) => r.score > 0.6).sort((a, b) => b.score - a.score);

    if (results.length === 0) { 
      console.log("No results after filtering (score > 0.0)");
      return c.json({ error: "No relevant context found after filtering" }, 404);
    }

    // UPDATED: Dynamic prompt for explanations/definitions
    const context = results
      .slice(0, 5)  // Top 5 for brevity
      .map((r, i) => `Source ${i + 1} (score: ${r.score.toFixed(2)}, file: ${r.metadata.sourceFile || 'unknown'}):\n${r.metadata.text}`)
      .join("\n\n");

    // Detect query type for better prompting
    const isDefinition = query.toLowerCase().includes('define') || query.toLowerCase().includes('definition');
    const prompt = isDefinition
  ? `Provide a concise definition of "${query}" using ONLY the following context, prioritizing higher-scoring sources. If no clear definition exists, state: "No clear definition found in context." Start directly with the definition.\n\nContext:\n${context}\n\nDefinition:`
  : `Explain "${query}" using ONLY the following context from your documents, prioritizing higher-scoring sources. Structure clearly but concisely. If insufficient, note briefly and suggest 2-3 related topics. Start directly with the explanation.\n\nContext:\n${context}\n\nExplanation:`;

    console.log("Sending prompt to HF:", prompt.slice(0, 200) + "...");  // Log truncated for readability
    const answer = await chatWithHF(prompt);

    // ADDED: Log full answer for debugging
    console.log("Full LLM response:", answer);

    return c.json({
      response: answer,
      sources: results.slice(0, 5).map((r) => ({
        id: r.id,
        text: r.metadata.text.slice(0, 100) + "...",
        score: r.score,
        sourceFile: r.metadata.sourceFile || 'unknown',
      })),
    });
  } catch (err: unknown) {
    console.error("Chat endpoint error:", err);
    const errorMessage = err instanceof Error ? err.message : String(err);
    return c.json({ error: "Internal server error", details: errorMessage }, 500);
  }
});
// Health check
app.get("/health", (c) => c.json({ ok: true }));

// Startup
console.log("🚀 RAG server running on http://localhost:8080");
Deno.serve({ port: 8080 }, app.fetch);