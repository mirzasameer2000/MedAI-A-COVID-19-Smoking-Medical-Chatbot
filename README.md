# MedAI: COVID-19 & Smoking Medical Chatbot

MedAI is a **ChatGPT-style medical research assistant** built in **Google Colab** using **LlamaIndex**, **Hugging Face models**, and **Gradio**.  
It performs **semantic search (RAG)** over CORD-19 dataset metadata and generates concise, factual responses.

> ⚠️ **Medical Disclaimer:** MedAI is for **educational and research** use only. It does **not** provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---

## Features

- 🔎 **Semantic search (RAG)** over CORD-19 dataset metadata (via KaggleHub)
- 🧠 **LLM answering** using a lightweight open-source model (Falcon-7B-Instruct in 4-bit quantization)
- 🧩 Text is chunked and indexed for retrieval-based responses
- 💬 **Gradio ChatInterface** for an interactive chatbot UI
- 💾 **Persistent vector index** saved to disk (`storage/`) to avoid re-indexing each run

---

## How It Works (Pipeline)

1. Download CORD-19 metadata from KaggleHub  
2. Extract the text field (descriptions), clean missing values  
3. Chunk text into **150-word** segments  
4. Create embeddings using **SentenceTransformers MiniLM**  
5. Build a **VectorStoreIndex** (LlamaIndex) and persist it to `storage/`  
6. Run a chat engine in “context” mode so answers are grounded in retrieved chunks  
7. Serve a Gradio web UI for chatting

---

## Tech Stack

- **RAG / Indexing:** LlamaIndex
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** `tiiuae/falcon-7b-instruct` (4-bit via bitsandbytes)
- **Dataset:** CORD-19 metadata (via KaggleHub)
- **UI:** Gradio

---

## Project Files

Typical repository contents:

- `MedAI.ipynb` — main Colab notebook (backend + frontend)
- `medai.py` — script version/export of the notebook
- `Requirments.txt`  — dependencies 
- `bandicam...mp4` — demo video 

---

## Run on Google Colab (Recommended)

1. Open `MedAI.ipynb` in Google Colab
2. Ensure GPU is enabled:
   - **Runtime → Change runtime type → GPU**
3. Run each cell **in order**
4. Gradio will start and provide a public `share=True` link

> The first run may take time because the model and dataset download + indexing happens once.
> Next runs reuse `storage/` if it exists.

---

## Run Locally (Optional)

> Local execution is heavier. A CUDA GPU is strongly recommended.

### 1) Create and activate environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
