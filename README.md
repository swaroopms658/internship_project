Hotel Booking Analytics with RAG

This project implements a Retrieval-Augmented Generation (RAG) pipeline for analyzing hotel booking data using FAISS for similarity search and a transformer model for question answering.

Project Structure

data processing.py: Preprocesses hotel booking data, generates sentence embeddings, and stores them in a FAISS index.

embedding_store.py: Loads the FAISS index and provides a search function for querying similar records.

evaluate_rag.py: Evaluates the RAG model using test questions and computes accuracy, precision, recall, and F1-score.

generate_faiss.py: Generates and saves a FAISS index from the hotel booking dataset.

main.py: Implements a FastAPI server to expose the RAG-based question-answering API.

rag.py: Handles retrieval from FAISS and answer generation using GPT-Neo.

Setup

1. Clone the Repository

git clone https://github.com/swaroopms658/internship_project
cd internship_project

2. Install Dependencies

pip install -r requirements.txt

3. Prepare Data

Ensure your dataset is available at data/hotel_booking.csv. Run the following to preprocess and create the FAISS index:

python generate_faiss.py

4. Run the API Server

uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000/.

API Usage

Ask a Question

Endpoint: POST /ask/

Request Body:

{
  "question": "What is the most booked hotel?"
}

Response:

{
  "question": "What is the most booked hotel?",
  "answer": "The most booked hotel is XYZ Hotel."
}

Evaluation

Run evaluate_rag.py to measure model performance:

python evaluate_rag.py

Future Improvements

Enhance retrieval with better embeddings.

Improve response generation with more powerful models.

Add support for multilingual queries.


