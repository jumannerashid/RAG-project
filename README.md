# RAG Project (Retrieval-Augmented Generation)

A backend service implementing **Retrieval-Augmented Generation (RAG)** using **Deno**.  
This project allows querying `.txt` and `.md` documents to generate AI-driven responses based on the most relevant context. It is deployable locally or on **Deno Deploy** for cloud access.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Architecture](#architecture)  
3. [Features](#features)  
4. [Documents](#documents)  
5. [Requirements](#requirements)  
6. [Installation](#installation)  
7. [Running the Project Locally](#running-the-project-locally)  
8. [Deployment](#deployment)  
9. [API Endpoint](#api-endpoint)  
10. [Body Parameters](#body-parameters)  
11. [Example Usage](#example-usage)  
12. [Troubleshooting](#troubleshooting)  
13. [Contributing](#contributing)  
14. [License](#license)  

---

## Project Overview

The **RAG backend** is designed to provide contextual answers from a set of documents. It combines **document retrieval** with **large language model (LLM) generation**, making it capable of answering questions even if the information is spread across multiple files.

Key use cases:

- Academic research (summarizing and querying lecture notes)  
- Financial document analysis (e.g., credit scoring, loan factors)  
- Knowledge base querying for internal company documentation  

The backend reads documents (`.txt` and `.md`), indexes them, and exposes a **REST API** endpoint for queries.

**Deployed backend URL:**  
[https://rag-project-5pj54jem3m1f.jumannerashid.deno.net](https://rag-project-5pj54jem3m1f.jumannerashid.deno.net)

---

## Architecture

┌─────────────┐
│ User Query │
└─────┬───────┘
│ POST /chat { "query": "..." }
▼
┌─────────────┐
│ RAG Server │
│ (Deno) │
└─────┬───────┘
│
▼
┌─────────────┐
│ Document │
│ Ingestion │
│ (.txt/.md) │
└─────┬───────┘
│
▼
┌─────────────┐
│ Retrieval │
│ (Vector DB) │
└─────┬───────┘
│
▼
┌─────────────┐
│ LLM │
│ (Generative │
│ Model) │
└─────┬───────┘
│
▼
┌─────────────┐
│ JSON Output │
│ { "answer": "..."} │
└─────────────┘

**Flow:**  
1. User sends a `query` to `/chat`.  
2. Backend retrieves relevant sections from `.txt` and `.md` documents.  
3. LLM generates a context-aware response.  
4. Backend returns JSON containing the answer.

---

## Features

- Indexing and searching `.txt` and `.md` files  
- Query endpoint for programmatic access (`/chat`)  
- Generates responses using relevant context  
- Fully deployable on **Deno Deploy** or locally  
- Easy integration with frontend apps or other APIs  

---

## Documents

The backend works with `.txt` and `.md` files.  
**Folder structure such as:**

/data
Bss.md
DCRT.md
kyambile 2018.md.md
udsm.md
docs1.txt
docs2.txt


- All documents must be placed in a folder readable by the backend (default: `data/`)  
- Supports multiple file types, `.txt` and `.md`  
- Text is indexed and vectorized for retrieval  

> Make sure the documents contain the information you expect to query, otherwise the response may be `"No relevant context found"`.

---

## Requirements

- **Deno >= 1.x**  
- `.txt` or `.md` documents  
- Internet connection for LLM API (if applicable)  
- Optional: environment variables for API keys  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jumannerashid/rag-project.git
cd rag-project

### Running the Project Locally

Run the backend:

deno run --allow-net --allow-read --allow-env main.ts


Permissions explained:

--allow-net → allow server network access

--allow-read → read local documents

--allow-env → read environment variables (e.g., API keys)

The server will print:

Listening on http://localhost:8080

Deployment

The project can be deployed on Deno Deploy for public access.

Example deployment URL:
https://rag-project-5pj54jem3m1f.jumannerashid.deno.net

API Endpoint

Endpoint: /chat
Method: POST
Headers: Content-Type: application/json

Local URL: http://localhost:8080/chat
Deployed URL: https://rag-project-5pj54jem3m1f.jumannerashid.deno.net/chat

Body Parameters
Parameter	Type	    Required	Description
query	    string	  Yes     	Question or prompt to search the documents

Example Usage
PowerShell
(Invoke-WebRequest `
  -Uri "https://rag-project-5pj54jem3m1f.jumannerashid.deno.net/chat" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{ "query": "Explain how credit scoring works and the factors that affect it." }').Content

cURL
curl -X POST https://rag-project-5pj54jem3m1f.jumannerashid.deno.net/chat \
-H "Content-Type: application/json" \
-d '{"query":"Explain how credit scoring works and the factors that affect it."}'

Example Queries

"What is credit scoring?"

"List the main factors affecting credit scores"

"Summarize the introduction section of the documents"

"Explain how financial documents are structured"

Troubleshooting
Issue	Solution
"No relevant context found"	Make sure documents contain the topic being queried. Check folder structure.
"Unexpected end of JSON input"	Ensure the POST body is valid JSON with "query" key.
Server not starting	Make sure Deno is installed and you are in the correct directory. Use --allow-net --allow-read --allow-env.
Contributing

Fork the repository

Create a new branch (feature/my-feature)

Make changes

Submit a Pull Request
License

MIT License © Jumanne Rashid

