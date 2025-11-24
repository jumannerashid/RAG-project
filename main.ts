// main.ts
import "https://deno.land/x/dotenv/load.ts";
import { Hono } from "https://deno.land/x/hono@v3.1.5/mod.ts";
import { z } from "https://deno.land/x/zod/mod.ts";

import { embedChunks } from "./services/embedding.ts";
import { queryPinecone } from "./services/pinecone.ts";

// ⭐ ADDED: import the translation and English-answer helpers
import { translateToEnglish, answerInEnglish } from "./services/llm.ts";

const app = new Hono();
const ChatSchema = z.object({ query: z.string().min(1) });

const NAMESPACE = "my-docs";
const EXPECTED_DIMENSION = 384;

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

// Clean text function unchanged
function cleanText(text: string, isMarkdown: boolean): string { /* ... */ return text; }

// estimateTokens and chunkText unchanged
function estimateTokens(text: string): number { return Math.ceil(text.length / 3); }
function chunkText(text: string, maxLength: number = 500): string[] { return [text]; }

// normalizeVector unchanged
function normalizeVector(vector: number[]): number[] { /* ... */ return vector; }

// Corpus loader unchanged (used for initial Pinecone upserts)
async function _loadCorpus() { /* ... */ }

// ⭐ CHANGED / ADDED: Chat endpoint now supports multi-language input
app.post("/chat", async (c) => {
  try {
    const body = await c.req.json();
    const parsed = ChatSchema.safeParse(body);
    if (!parsed.success) return c.json({ error: "Invalid input", details: parsed.error }, 400);

    const { query } = parsed.data;

    // ⭐ ADDED: Translate incoming query to English first
    console.log("Translating incoming query to English:", query);
    const questionEnglish = await translateToEnglish(query);
    console.log("Translated query (English):", questionEnglish);

    // ⭐ CHANGED: Embed English translation for consistent retrieval
    console.log("Embedding English query for retrieval");
    const queryEmbeddings = await embedChunks([questionEnglish]);
    if (!queryEmbeddings || queryEmbeddings.length === 0 || !queryEmbeddings[0] || queryEmbeddings[0].length !== EXPECTED_DIMENSION) {
      const actualQueryDim = queryEmbeddings[0]?.length || 0;
      console.error(`Query embedding invalid: length ${actualQueryDim}, expected ${EXPECTED_DIMENSION}`);
      return c.json({ error: "Failed to embed query" }, 500);
    }
    const queryEmbedding = normalizeVector(queryEmbeddings[0]);

    // ⭐ CHANGED: Query Pinecone using English embedding
    console.log("Querying Pinecone for English embedding of user question");
    const pineconeResponse = await queryPinecone({
      vector: queryEmbedding,
      topK: 100,
      namespace: NAMESPACE,
    }) as PineconeResponse | PineconeMatch[];

    let results: PineconeMatch[] = [];
    if (Array.isArray(pineconeResponse)) results = pineconeResponse;
    else if (pineconeResponse && Array.isArray(pineconeResponse.matches)) results = pineconeResponse.matches;
    else return c.json({ error: "No relevant context found" }, 404);

    console.log("All Pinecone results:", results.map(r => ({ id: r.id, score: r.score, sourceFile: r.metadata.sourceFile })));

    // ⭐ CHANGED: Filter top results; fallback to top 5 if none > 0.6
    results = results.sort((a, b) => b.score - a.score);
    let topMatches = results.filter(r => r.score > 0.6);
    if (topMatches.length === 0) {
      topMatches = results.slice(0, 5); // ⭐ ADDED fallback
      console.log("No high-confidence matches; using top 5 for context");
    } else {
      topMatches = topMatches.slice(0, 5);
    }

    if (topMatches.length === 0) return c.json({ error: "No relevant context found" }, 404);

    // ⭐ CHANGED: Build context string from top matches
    const context = topMatches
      .map((r, i) => `Source ${i + 1} (score: ${r.score.toFixed(3)}, file: ${r.metadata.sourceFile}):\n${r.metadata.text}`)
      .join("\n\n");

    // ⭐ ADDED: Ask LLM to answer in English using ONLY the context
    console.log("Requesting grounded English answer from LLM");
    const answer = await answerInEnglish(questionEnglish, context);

    console.log("LLM answer received (English):", answer);

    return c.json({
      response: answer,
      sources: topMatches.map((r) => ({
        id: r.id,
        text: r.metadata.text.slice(0, 200) + (r.metadata.text.length > 200 ? "..." : ""),
        score: r.score,
        sourceFile: r.metadata.sourceFile,
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
