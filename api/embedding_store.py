import faiss
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# Paths
csv_path = "D:/internship_project/data/hotel_booking.csv"
index_path = "D:/internship_project/data/hotel_booking.index"

# Load dataset and validate columns
df = pd.read_csv(csv_path)
print(f"Loaded dataset with columns: {df.columns.tolist()}")

# Find room-related columns
room_columns = [col for col in df.columns if "room" in col.lower()]
print(f"Room-related columns found: {room_columns}")

# Basic required columns - adjust based on available columns
required_columns = {"hotel", "country"}
available_required = required_columns.intersection(set(df.columns))
print(f"Available required columns: {available_required}")

if not available_required:
    raise ValueError(f"CSV must contain at least some of these columns: {required_columns}")

# Clean and concatenate text data
df.fillna("", inplace=True)

# Dynamically create text representation based on available columns
text_parts = []
if "hotel" in df.columns:
    text_parts.append(df["hotel"].astype(str))
if "arrival_date_month" in df.columns:
    text_parts.append(df["arrival_date_month"].astype(str))
if "country" in df.columns:
    text_parts.append(df["country"].astype(str))
if "adr" in df.columns:
    text_parts.append("Price: $" + df["adr"].astype(str))
# Add room type info if available
for col in room_columns:
    text_parts.append(col + ": " + df[col].astype(str))

# Join all parts with spaces
text_data = [" ".join(parts) for parts in zip(*text_parts)]

# Load embedding model and FAISS index
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
if not os.path.exists(index_path):
    raise FileNotFoundError(f"FAISS index not found at {index_path}. Run `generate_faiss.py` first.")
index = faiss.read_index(index_path)

def search_faiss(query, top_k=3):
    """Search the FAISS index for similar records."""
    if index is None:
        return ["FAISS index not loaded properly."]
    
    query_vector = embed_model.encode([query]).astype(np.float32)
    
    distances, indices = index.search(query_vector, top_k)
    
    # Ensure valid index range
    results = [text_data[i] for i in indices[0] if 0 <= i < len(text_data)]
    
    if not results:
        return ["No relevant records found."]
    
    return results