// services/llm.ts
//import "https://deno.land/x/dotenv/load.ts";
const GROQ_API_KEY = Deno.env.get("GROQ_API_KEY");

if (!GROQ_API_KEY) {
  throw new Error("GROQ_API_KEY is not set in environment variables");
}

// =========================
//  LOW-LEVEL RAW API CALL
// =========================
export async function chatWithHF(prompt: string, retries = 3): Promise<string> {
  if (!prompt || typeof prompt !== "string" || prompt.trim() === "") {
    throw new Error("Invalid input: prompt must be a non-empty string");
  }

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      console.log(
        `Sending chat request to Groq model: llama-3.3-70b-versatile (attempt ${attempt}/${retries})`
      );

      const response = await fetch(
        "https://api.groq.com/openai/v1/chat/completions",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${GROQ_API_KEY}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: "llama-3.3-70b-versatile",
            messages: [{ role: "user", content: prompt }],
            max_tokens: 500,
            temperature: 0.1,
          }),
        }
      );

      if (!response.ok) {
        const text = await response.text();
        throw new Error(
          `Groq API error: ${response.status} ${response.statusText} – ${text}`
        );
      }

      const data = await response.json();
      const generatedText =
        data.choices?.[0]?.message?.content?.trim();

      if (!generatedText) {
        throw new Error("No generated text in Groq response");
      }

      console.log("Successfully received chat response");
      return generatedText;
    } catch (err: unknown) {
      console.error(`Chat request failed (attempt ${attempt}):`, err);

      if (attempt === retries) {
        throw new Error(
          `Chat request failed after ${retries} attempts: ${
            err instanceof Error ? err.message : String(err)
          }`
        );
      }

      await new Promise((resolve) =>
        setTimeout(resolve, 1000 * attempt)
      );
    }
  }

  throw new Error("Chat request failed after all retries");
}

// =========================
//   TRANSLATION (Sw→En)
// =========================
export async function translateToEnglish(text: string): Promise<string> {
  const prompt = `
Translate the following text from Swahili (or any language) to clear English.
Return ONLY the English translation, no explanation.

TEXT:
${text}
  `;

  return await chatWithHF(prompt);
}

// =========================
//  FINAL ANSWER (English)
// =========================
export async function answerInEnglish(question: string, context: string): Promise<string> {
  const prompt = `
You are a helpful assistant. 
ALWAYS respond in English, even if the user question was not in English.

USER QUESTION (English):
${question}

CONTEXT (use ONLY this information):
${context}

Provide a helpful answer in English.
  `;

  return await chatWithHF(prompt);
}
