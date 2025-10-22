// services/embedding.ts
import { load } from "jsr:@std/dotenv";

import { HfInference } from "https://esm.sh/@huggingface/inference@4.11.3/denonext/inference.mjs";

const localEnv = await load({
  export: true,
});
console.log('localEnv ', localEnv, Deno.env.get("HF_API_KEY"));

const HF_API_KEY = localEnv.HF_API_KEY;
const PINECONE_API_KEY = localEnv.PINECONE_API_KEY;
const PINECONE_ENVIRONMENT = localEnv.PINECONE_ENVIRONMENT;
const PINECONE_INDEX = localEnv.PINECONE_INDEX;

if (!HF_API_KEY) throw new Error("HF_API_KEY is not set in environment variables");
if (!PINECONE_API_KEY) throw new Error("PINECONE_API_KEY is not set in environment variables");
if (!PINECONE_ENVIRONMENT) throw new Error("PINECONE_ENVIRONMENT is not set in environment variables");
if (!PINECONE_INDEX) throw new Error("PINECONE_INDEX is not set in environment variables");
const client = new HfInference(HF_API_KEY);

export async function embedChunks(chunks: string[], retries = 3): Promise<number[][]> {
  if (!chunks || !Array.isArray(chunks) || chunks.length === 0) {
    throw new Error("Invalid input: chunks array is empty or not an array");
  }
  if (!chunks.every(chunk => typeof chunk === "string" && chunk.trim() !== "")) {
    throw new Error("Invalid input: all chunks must be non-empty strings");
  }

  // Normalize chunks to remove extra whitespace
  const normalizedChunks = chunks.map(chunk => chunk.replace(/\s+/g, " ").trim());
  console.log("Embedding documents:", normalizedChunks.length);
  console.log("Input chunks (first 100 chars each):", normalizedChunks.map(c => c.substring(0, 100) + "..."));

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      console.log(`Embedding attempt ${attempt}/${retries}`);
      const response = await client.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: normalizedChunks, // Batch process all chunks
      });

      // Validate response: should be an array of 384-dimensional embeddings
      if (
        !Array.isArray(response) ||
        response.length !== normalizedChunks.length ||
        !response.every(arr => Array.isArray(arr) && arr.length === 384 && arr.every(n => typeof n === "number"))
      ) {
        throw new Error("Invalid embedding response from Hugging Face API");
      }

      console.log("✅ Successfully embedded documents");
      return response as number[][];
    } catch (error) {
      console.error(`Embedding error (attempt ${attempt}):`, error);
      console.error("Error details:", error.message, error.stack);
      if (attempt === retries) throw new Error(`HF embedding error: ${error.message}`);
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
  }

  throw new Error("Embedding failed after all retries");
}