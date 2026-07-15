from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from mistralai.client import Mistral

import faiss
import numpy as np
import re
import os

# ====================================
# PDF LOADING
# ====================================

pdf_path = "The_Adventures_of_Sherlock_Holmes.pdf"

reader = PdfReader(pdf_path)

print("Total Pages:", len(reader.pages))

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text

print("Total Characters:", len(text))

# ====================================
# TEXT CLEANING
# ====================================

start = text.find("Adventure F")

if start != -1:
    clean_text = text[start:]
else:
    clean_text = text

clean_text = re.sub(
    r'[^\w\s.,!?;:\'\"()\-]',
    ' ',
    clean_text
)

clean_text = re.sub(
    r'\s+',
    ' ',
    clean_text
)

clean_text = clean_text.strip()

print("Clean Text Length:", len(clean_text))

# ====================================
# CHUNKING
# ====================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(clean_text)

print("Total Chunks:", len(chunks))

# ====================================
# EMBEDDING MODEL
# ====================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDINGS_FILE = "embeddings.npy"
FAISS_FILE = "faiss_index.bin"

# ====================================
# LOAD OR CREATE EMBEDDINGS
# ====================================

if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_FILE):

    print("Loading cached embeddings...")

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    index = faiss.read_index(
        FAISS_FILE
    )

else:

    print("Creating embeddings...")

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    faiss.write_index(
        index,
        FAISS_FILE
    )

    print("Embeddings saved!")

print("FAISS index ready!")

# ====================================
# MISTRAL CLIENT
# ====================================

from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

api_key = os.getenv("mistral_key")

client = Mistral(
    api_key=api_key
)

print("Mistral client ready!")

# ====================================
# RAG FUNCTION
# ====================================

def rag_pipeline(question, conversation):

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        k=5
    )

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append(
            chunks[idx]
        )

    context = "\n\n".join(
        retrieved_chunks
    )

    # ====================================
    # CHAT MEMORY
    # ====================================

    chat_history = ""

    for msg in conversation[-10:]:

        chat_history += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    # ====================================
    # PROMPT
    # ====================================

    prompt = f"""
You are a Story Book Assistant.

Previous Conversation:
{chat_history}

Story Context:
{context}

Current Question:
{question}

Instructions:

- Use the story context when answering.
- Use previous conversation to resolve references.
- If the user says:
  he, she, him, her, his, their,
  determine who they refer to from chat history.
- Do not answer for the question whose name is outside the context.
- Keep answers clear and concise.

Answer:
"""

    response = client.chat.complete(

        model="mistral-small-latest",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return answer


# ====================================
# TERMINAL TESTING
# ====================================

if __name__ == "__main__":

    conversation = []

    while True:

        question = input(
            "\nAsk Question (type exit to quit): "
        )

        if question.lower() == "exit":
            break

        conversation.append(
            {
                "role": "user",
                "content": question
            }
        )

        answer = rag_pipeline(
            question,
            conversation
        )

        conversation.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        print("\nAnswer:\n")
        print(answer)