import { load } from "https://deno.land/std@0.170.0/dotenv/mod.ts";

const env = await load();
const PINECONE_API_KEY = Deno.env.get("PINECONE_API_KEY");
const PINECONE_HOST = Deno.env.get("PINECONE_HOST") || "us-west1-gcp";
const PINECONE_INDEX = Deno.env.get("PINECONE_INDEX") || "jumanne";

if (!PINECONE_API_KEY || !PINECONE_HOST || !PINECONE_INDEX) {
  console.error("Missing Pinecone environment variables:");
  console.error(`PINECONE_API_KEY: ${PINECONE_API_KEY ? "Set" : "Missing"}`);
  console.error(`PINECONE_ENVIRONMENT: ${PINECONE_HOST ? PINECONE_HOST : "Missing"}`);
  console.error(`PINECONE_INDEX: ${PINECONE_INDEX ? PINECONE_INDEX : "Missing"}`);
  throw new Error("Missing required Pinecone environment variables. Please check .env file.");
}

//const PINECONE_HOST = `${PINECONE_HOST}.pinecone.io`;

interface Vector {
  id: string;
  vector: number[];
  metadata: { text: string };
}

export async function upsertToPinecone(
  ids: string[],
  vector: Vector,
  namespace: string,
  upsertMetadata = false,
): Promise<void> {
  try {
    console.log("Upserting to Pinecone URL:", PINECONE_HOST);
    const body = {
      vectors: [
        {
          id: vector.id,
          values: vector.vector,
          metadata: upsertMetadata ? vector.metadata : undefined,
        },
      ],
      namespace,
    };

    const response = await fetch(`${PINECONE_HOST}/vectors/upsert`, {
      method: "POST",
      headers: {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Pinecone upsert failed: ${response.status} ${text}`);
    }

    console.log(`✅ Upserted vector ${vector.id} to Pinecone namespace ${namespace}`);
  } catch (err) {
    console.error("Pinecone upsert error:", err);
    throw err;
  }
}

export async function queryPinecone(query: {
  vector: number[];
  topK: number;
  namespace: string;
}): Promise<{ id: string; score: number; metadata: { text: string } }[]> {
  try {
    console.log("Querying Pinecone URL:", PINECONE_HOST);
    const body = {
      vector: query.vector,
      topK: query.topK,
      includeMetadata: true,
      namespace: query.namespace,
    };

    const response = await fetch(`${PINECONE_HOST}/query`, {
      method: "POST",
      headers: {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Pinecone query failed: ${response.status} ${text}`);
    }

    const result = await response.json();
    const matches = result.matches || [];
    console.log(`✅ Queried Pinecone, found ${matches.length} matches`);

    return matches.map((match: { id: string; score: number; metadata: { text: string } }) => ({
      id: match.id,
      score: match.score,
      metadata: match.metadata,
    }));
  } catch (err) {
    console.error("Pinecone query error:", err);
    throw err;
  }
}