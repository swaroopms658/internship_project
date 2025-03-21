import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load the FAISS index and the sentence-transformer model
INDEX_PATH = "data/hotel_booking.index"
FAISS_INDEX = None
EMBEDDING_MODEL = None
GENERATOR = None

# Initialize models
def initialize_models():
    global FAISS_INDEX, EMBEDDING_MODEL, GENERATOR

    # Load FAISS index
    FAISS_INDEX = faiss.read_index(INDEX_PATH)

    # Load embedding model
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize GPT-Neo pipeline
    GENERATOR = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Retrieve relevant context using FAISS
def retrieve_context(question, top_k=3):
    global FAISS_INDEX, EMBEDDING_MODEL

    # Encode the question
    question_embedding = EMBEDDING_MODEL.encode(question).reshape(1, -1)

    # Search for top_k similar contexts in the FAISS index
    distances, indices = FAISS_INDEX.search(np.array(question_embedding, dtype="float32"), top_k)

    # Fetch the corresponding contexts
    contexts = []
    with open("data/processed_hotel_booking.csv", "r") as f:
        lines = f.readlines()
        for idx in indices[0]:
            contexts.append(lines[idx].strip())

    return contexts

# Generate an answer using GPT-Neo
def generate_answer(question, contexts):
    global GENERATOR

    # Combine the question and contexts
    input_text = "Context: " + " ".join(contexts) + f"\n\nQuestion: {question}\n\nAnswer:"

    # Generate text using the pipeline
    response = GENERATOR(input_text, do_sample=True, max_length=512, min_length=50)
    return response[0]["generated_text"]

# Main function for retrieval and generation
def get_answer(question):
    # Retrieve context
    contexts = retrieve_context(question)

    # Generate answer
    answer = generate_answer(question, contexts)

    return answer

# Initialize models when the script is loaded
initialize_models()
