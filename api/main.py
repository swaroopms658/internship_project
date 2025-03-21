from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.rag import get_answer

app = FastAPI()

# Request model for question
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    try:
        # Retrieve the answer using the RAG pipeline
        answer = get_answer(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG-based Hotel Booking Analytics API!"}
