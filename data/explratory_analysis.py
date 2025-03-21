import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import faiss
import os

# Paths
processed_csv_path = r"D:/internship_project/data/processed_hotel_booking.csv"
index_path = r"D:/internship_project/data/hotel_booking.index"

# Load preprocessed data
print("Loading preprocessed dataset...")
try:
    df = pd.read_csv(processed_csv_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"Processed dataset not found at: {processed_csv_path}")

# Step 1: Exploratory Data Analysis (EDA)

# General statistics
print("\nGeneral Dataset Statistics:")
print(df.describe(include="all"))

# Cancellation rates
if "is_canceled" in df.columns:
    print("\nCalculating cancellation rates...")
    cancel_rate = df["is_canceled"].value_counts(normalize=True) * 100
    print(f"Cancellation Rate:\n{cancel_rate}")

# Plot cancellation rates
plt.figure(figsize=(6, 4))
sns.barplot(x=cancel_rate.index, y=cancel_rate.values, palette="Blues_r")
plt.xticks([0, 1], ["Not Canceled", "Canceled"])
plt.title("Booking Cancellation Rate")
plt.ylabel("Percentage")
plt.show()

# Monthly booking trends
if "arrival_date_month" in df.columns:
    print("\nVisualizing monthly booking trends...")
    month_order = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    df["arrival_date_month"] = df["arrival_date_month"].astype(str)
    monthly_trends = df["arrival_date_month"].value_counts().reindex(month_order, fill_value=0)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=monthly_trends.index, y=monthly_trends.values, palette="coolwarm")
    plt.title("Monthly Booking Trends")
    plt.xlabel("Month")
    plt.ylabel("Number of Bookings")
    plt.show()

# Average length of stay by market segment
if "market_segment" in df.columns and "stays_in_weekend_nights" in df.columns and "stays_in_week_nights" in df.columns:
    print("\nAnalyzing average stay length by market segment...")
    df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    avg_stay_by_segment = df.groupby("market_segment")["total_stay"].mean().sort_values()
    print(avg_stay_by_segment)

    # Plot average stay length
    plt.figure(figsize=(8, 5))
    sns.barplot(x=avg_stay_by_segment.values, y=avg_stay_by_segment.index, palette="viridis")
    plt.title("Average Stay Length by Market Segment")
    plt.xlabel("Average Stay Length (Nights)")
    plt.ylabel("Market Segment")
    plt.show()

# Step 2: Analytical Query Handling with Embeddings

# Load FAISS index
print("\nLoading FAISS index for query handling...")
try:
    index = faiss.read_index(index_path)
    print("FAISS index loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"FAISS index not found at: {index_path}")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define function for analytical query handling
def query_analytics(query, df, index, model, column="description"):
    print(f"\nProcessing query: {query}")
    
    # Generate embedding for query
    query_embedding = model.encode([query]).astype("float32")
    
    # Search FAISS index
    D, I = index.search(query_embedding, k=5)  # Top-5 results
    print(f"Top matches (distances): {D.flatten()}")
    
    # Retrieve corresponding rows
    results = df.iloc[I.flatten()]
    return results

# Example query
example_query = "What are the most common reasons for cancellations?"
results = query_analytics(example_query, df, index, embedding_model)
print("\nQuery Results:")
print(results[["is_canceled", "description"]])

# Step 3: Save Analysis Outputs

# Save updated dataset with total_stay column
output_path = r"D:/internship_project/data/updated_hotel_booking.csv"
print(f"\nSaving updated dataset to {output_path}...")
df.to_csv(output_path, index=False)

print("\nMilestone 2 completed successfully!")
