import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Paths
CSV_PATH = "D:/internship_project/data/hotel_booking.csv"
INDEX_PATH = "D:/internship_project/data/hotel_booking.index"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index():
    """Generate FAISS index from hotel booking dataset."""
    
    # Load dataset and log available columns
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset with columns: {df.columns.tolist()}")
    
    # Find room-related columns
    room_columns = [col for col in df.columns if "room" in col.lower()]
    print(f"Room-related columns found: {room_columns}")
    
    # More flexible column requirement checking
    important_columns = {"hotel", "country"}
    available_columns = important_columns.intersection(set(df.columns))
    
    if not available_columns:
        raise ValueError(f"CSV should contain at least some of these columns: {important_columns}")
    
    # Clean missing values
    df.fillna("", inplace=True)
    
    # Dynamically build text representation based on available columns
    text_parts = []
    
    if "hotel" in df.columns:
        text_parts.append(df["hotel"].astype(str))
    
    if "country" in df.columns:
        # Put more emphasis on country field for better retrieval
        text_parts.append("country: " + df["country"].astype(str) + " ")
        text_parts.append("in " + df["country"].astype(str))
    
    if "arrival_date_month" in df.columns:
        text_parts.append("during " + df["arrival_date_month"].astype(str))
    
    if "adr" in df.columns:
        text_parts.append("with price: $" + df["adr"].astype(str))
    
    # Add room type info if available
    for col in room_columns:
        if col in df.columns:
            text_parts.append(f"{col}: " + df[col].astype(str))
    
    # Join with appropriate separators
    texts = []
    for row_parts in zip(*text_parts):
        texts.append(" ".join(row_parts))
    
    # Save the text representations for debugging
    debug_file = os.path.join(os.path.dirname(INDEX_PATH), "text_representations.txt")
    with open(debug_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts[:100]):  # Save first 100 for inspection
            f.write(f"{i}: {text}\n")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    
    print(f"Creating embeddings for {len(texts)} records...")
    
    # Convert text to embeddings
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)  # Ensure correct data type
    
    # Create FAISS index
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to index")
    
    # Save index to file system
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index to {INDEX_PATH}")
    
if __name__ == "__main__":
    create_faiss_index()