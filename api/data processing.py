import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# Paths
csv_path = r"D:/internship_project/data/hotel_booking.csv"
index_path = r"D:/internship_project/data/hotel_booking.index"
processed_csv_path = r"D:/internship_project/data/processed_hotel_booking.csv"

# Load dataset
print("Loading dataset...")
try:
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

# Step 1: Inspect and clean data
print("Inspecting dataset...")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample Data:\n{df.head()}")

# Handle missing values
print("Handling missing values...")
df.fillna("", inplace=True)

# Normalize country codes (convert to uppercase and trim spaces)
if "country" in df.columns:
    print("Normalizing country codes...")
    df["country"] = df["country"].str.strip().str.upper()

# Normalize other categorical columns if needed
# Example: Arrival month
if "arrival_date_month" in df.columns:
    print("Normalizing month names...")
    month_map = {
        "January": "01", "February": "02", "March": "03",
        "April": "04", "May": "05", "June": "06",
        "July": "07", "August": "08", "September": "09",
        "October": "10", "November": "11", "December": "12"
    }
    df["arrival_date_month"] = df["arrival_date_month"].map(month_map).fillna(df["arrival_date_month"])

# Step 2: Generate sentence embeddings
print("Generating sentence embeddings for data...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sentence_column = "description"  # Replace this with your relevant column for query matching

if sentence_column in df.columns:
    descriptions = df[sentence_column].astype(str).tolist()
    print(f"Generating embeddings for {len(descriptions)} rows...")
    embeddings = embedding_model.encode(descriptions)
else:
    print(f"Column '{sentence_column}' not found. Skipping embeddings...")
    embeddings = None

# Step 3: Build and save FAISS index
if embeddings is not None:
    print("Building FAISS index...")
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
    index.add(np.array(embeddings).astype("float32"))
    
    # Save the index
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
else:
    print("No embeddings generated. Skipping FAISS index creation...")

# Step 4: Save processed dataset
print(f"Saving processed dataset to {processed_csv_path}...")
df.to_csv(processed_csv_path, index=False)

print("Data preprocessing completed!")
