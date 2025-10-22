// services/llm.ts
import { load } from "https://deno.land/std@0.170.0/dotenv/mod.ts";

const env = await load();
const GROQ_API_KEY = Deno.env.get("GROQ_API_KEY");

if (!GROQ_API_KEY) throw new Error("GROQ_API_KEY is not set in environment variables");

export async function chatWithHF(prompt: string, retries = 3): Promise<string> {
  if (!prompt || typeof prompt !== "string" || prompt.trim() === "") {
    throw new Error("Invalid input: prompt must be a non-empty string");
  }

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      console.log(`Sending chat request to Groq model: llama-3.3-70b-versatile (attempt ${attempt}/${retries})`);
      const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "llama-3.3-70b-versatile",
          messages: [{ role: "user", content: prompt }],
          max_tokens: 500,
          temperature: 0.7,
        }),
      });

      if (!response.ok) {
        throw new Error(`Groq API error: ${response.statusText}`);
      }

      const data = await response.json();
      const generatedText = data.choices[0]?.message?.content || "";

      if (!generatedText) {
        throw new Error("No generated text in Groq response");
      }

      console.log("✅ Successfully received chat response");
      return generatedText.trim();
    } catch (error) {
      console.error(`Chat request failed (attempt ${attempt}):`, error);
      if (attempt === retries) throw new Error(`Chat request failed: ${error.message}`);
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
  }

  throw new Error("Chat request failed after all retries");
}