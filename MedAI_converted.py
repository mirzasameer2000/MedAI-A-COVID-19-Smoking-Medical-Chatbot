#!/usr/bin/env python
# coding: utf-8

# # **BACK-END**
# 
# > Advanced Natural Language Processing
# 
# > Muhammad Sameer (XNV7SX)
# 
# 
# > University of Debrecen
# 
# 
# 
# 
# 
# 
# 
# 

# 
# **Install Requirements for Back-End**

# In[ ]:


# !pip install -q llama-index \
#    llama-index-embeddings-huggingface \
#    llama-index-llms-huggingface \
#    bitsandbytes \
#    kagglehub


# **Backend Setup**

# In[2]:

# %% [setup] install runtime deps into THIS kernel
import sys, subprocess

# Upgrade pip tooling first (avoids many Windows issues)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

# Core libs + LlamaIndex HuggingFace plugins
reqs = [
    "pandas",
    "tqdm",
    "kagglehub",
    "llama-index",
    "llama-index-embeddings-huggingface",
    "llama-index-llms-huggingface",
    # helpful extras
    "transformers",
    "sentence-transformers",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", *reqs])

print("✅ Dependencies installed in:", sys.executable)


import os
import pandas as pd
import kagglehub
from tqdm import tqdm

from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.memory import ChatMemoryBuffer


# **Load dataset**

# In[3]:


path = kagglehub.dataset_download(handle="googleai/dataset-metadata-for-cord19")
filename = os.path.join(path, os.listdir(path)[0])
df = pd.read_csv(filename)
df = df[df['description'].notnull()][['description']]


# **Chunk text into 150-word sections**

# In[4]:


chunks = []
chunk_size = 150
for text in tqdm(df['description']):
    words = str(text).split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(Document(text=chunk))


# **Set embedding model**

# In[ ]:


Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
)


# **Load or create vector index**

# In[6]:


persist_dir = "storage"
if not os.path.exists(persist_dir):
    print("🔧 Creating new index...")
    index = VectorStoreIndex.from_documents(chunks, show_progress=True)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    print("📂 Loading index from disk...")
    storage = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage)


# **Use falcon-7b**

# In[ ]:


from llama_index.llms.huggingface import HuggingFaceLLM

Settings.llm = HuggingFaceLLM(
    model_name="tiiuae/falcon-7b-instruct",
    tokenizer_name="tiiuae/falcon-7b-instruct",
    context_window=2048,
    max_new_tokens=256,
    device_map="auto",
    tokenizer_kwargs={"use_fast": True},
    model_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "trust_remote_code": True
    },
    generate_kwargs={"temperature": 0.7, "do_sample": True},
)


# **Create chat engine**

# In[8]:


chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=ChatMemoryBuffer.from_defaults(token_limit=2048),
    system_prompt="""
You are MedAI, a medical chatbot. Answer only the user's question.
Do not repeat it. Do not ask follow-up questions. Be concise and factual.
"""
)


# _________________________________________________________________________________________________________________________
# 

# ____________________________________________________________________________

# # **Front-END**

# **Install Requirements for Front-End**

# In[ ]:


# pip install gradio llama-index llama-index-embeddings-huggingface llama-index-llms-huggingface bitsandbytes --quiet


# In[10]:


import gradio as gr


# **Function to handle chat**

# In[11]:


def medai_chat(user_input, _):
    try:
        user_input = user_input.strip()
        response = chat_engine.chat(user_input)
        return response.response
    except Exception as e:
        return f"[ERROR] {str(e)}"


# **Gradio interface**

# In[12]:


chat_ui = gr.ChatInterface(
    fn=medai_chat,
    chatbot=gr.Chatbot(label="🩺 MedAI Assistant"),
    title="MedAI - COVID & Smoking Research Chatbot",
    description="Ask specific questions about COVID-19, smoking, and respiratory health. MedAI will respond concisely and factually.",
    theme="soft",
    examples=[
        "What are the risks of smoking during COVID-19?",
        "Can COVID-19 worsen lung conditions?",
        "How does smoking affect immune response?",
         "What are the similar effects on patients who smoke and those with COVID-19 in the human body?",
        "Can smoking increase the severity of COVID-19 symptoms?",
        "What organs are affected by smoking and COVID-19?"
    ]
)

chat_ui.launch(share=True)

