import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from api.rag import get_answer  # Import the RAG logic

# Load the test data
with open("data/test_data.json", "r") as f:
    data = json.load(f)

# Initialize lists for evaluation
true_answers = []
predicted_answers = []

# Iterate over test cases
for item in data:
    question = item["question"]
    expected_answer = item["expected_answer"]

    # Fetch the answer using the RAG logic
    predicted_answer = get_answer(question)

    true_answers.append(expected_answer)
    predicted_answers.append(predicted_answer)

# Calculate metrics
accuracy = accuracy_score(true_answers, predicted_answers)
precision = precision_score(true_answers, predicted_answers, average="micro")
recall = recall_score(true_answers, predicted_answers, average="micro")
f1 = f1_score(true_answers, predicted_answers, average="micro")

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
