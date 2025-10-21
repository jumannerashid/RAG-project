// main.ts
import "https://deno.land/x/dotenv/load.ts";
import { Hono } from "https://deno.land/x/hono@v3.1.5/mod.ts";
import { z } from "https://deno.land/x/zod/mod.ts";

import { embedChunks } from "./services/embedding.ts";
import { upsertToPinecone, queryPinecone } from "./services/pinecone.ts";
import { chatWithHF } from "./services/llm.ts";

const app = new Hono();
const ChatSchema = z.object({ query: z.string().min(1) });

const NAMESPACE = "my-docs";

// Enable CORS
app.use("*", async (c, next) => {
  c.header("Access-Control-Allow-Origin", "*");
  c.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  c.header("Access-Control-Allow-Headers", "Content-Type");
  if (c.req.method === "OPTIONS") return c.text("", 204);
  await next();
});

// Load all files from ./data and upsert to Pinecone
async function loadCorpus() {
  const dir = "./data";
  const files: { id: string; text: string }[] = [];
  try {
    for await (const entry of Deno.readDir(dir)) {
      if (entry.isFile && entry.name.endsWith(".txt")) {
        const text = await Deno.readTextFile(`${dir}/${entry.name}`);
        files.push({ id: entry.name, text });
      }
    }

    if (files.length === 0) {
      console.log("No documents found in ./data");
      return;
    }

    // Embed all texts
    console.log("Embedding documents:", files.map(f => f.id));
    const chunks = files.map(f => f.text);
    const rawEmbeddings = await embedChunks(chunks) as number[][];
    if (!rawEmbeddings || rawEmbeddings.length !== chunks.length) {
      throw new Error("Embedding failed: Invalid response from HuggingFace");
    }
    const embeddings = rawEmbeddings.map((vector, index) => ({
      id: files[index].id,
      vector,
      metadata: { text: files[index].text },
    }));

    // Upsert to Pinecone
    for (const embedding of embeddings) {
      console.log("Upserting to Pinecone:", embedding.id);
      await upsertToPinecone([embedding.id], embedding, NAMESPACE, true);
    }

    console.log(`✅ Loaded ${files.length} documents into Pinecone.`);
  } catch (err) {
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
    if (!queryEmbeddings || queryEmbeddings.length === 0) {
      return c.json({ error: "Failed to embed query" }, 500);
    }
    const queryEmbedding = queryEmbeddings[0];

    // Query Pinecone
    console.log("Querying Pinecone for:", query);
    const results = await queryPinecone({
      vector: queryEmbedding,
      topK: 3,
      namespace: NAMESPACE,
    });

    if (!results || results.length === 0) {
      return c.json({ error: "No relevant context found" }, 404);
    }

    // Build context
    const context = results.map((r: { metadata: { text: string } }) => `• ${r.metadata.text}`).join("\n");
    const prompt = `Answer based ONLY on the following:\n${context}\n\nUser: ${query}`;

    console.log("Sending prompt to HF:", prompt);
    const answer = await chatWithHF(prompt);

    return c.json({
      response: answer,
      sources: results.map((r: { id: unknown; metadata: { text: string }; score: unknown }) => ({
        id: r.id,
        text: r.metadata.text.slice(0, 100) + "...",
        score: r.score,
      })),
    });
  } catch (err) {
    console.error("Chat endpoint error:", err);
    return c.json({ error: "Internal server error", details: `Chat error: ${err.message}` }, 500); // Fixed error label
  }
});
// Health check
app.get("/health", (c) => c.json({ ok: true }));

// Startup
try {
 //  await loadCorpus();
  console.log("🚀 RAG server running on http://localhost:8080");
  Deno.serve({ port: 8080 }, app.fetch);
} catch (err) {
  console.error("Startup failed:", err);
  Deno.exit(1);
}