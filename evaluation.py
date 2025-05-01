
import os
import openai
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# === Step 1: Load your OpenAI API Key ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Step 2: Load Your Data ===
ground_truth_df = pd.read_excel('Bridge_Risk_Ground_Truth.xlsx')
predictions_df = pd.read_csv('risk_assessment2.csv')

# Combine 'Risk Factor' + 'Risk Detail' for better meaning
ground_truth_texts = (ground_truth_df['Risk Factor'] + " - " + ground_truth_df['Risk Description']).dropna().tolist()
predicted_texts = (predictions_df['Risk Factor'] + " - " + predictions_df['Risk Detail']).dropna().tolist()

# === Step 3: Define New Embedding Function for OpenAI v1.0+ ===
from openai import OpenAI

client = OpenAI()


def get_openai_embedding(text):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


# === Step 4: Generate Embeddings ===
print("Embedding ground truth risks...")
ground_truth_embeddings = [get_openai_embedding(text) for text in ground_truth_texts]

print("Embedding predicted risks...")
predicted_embeddings = [get_openai_embedding(text) for text in predicted_texts]

# === Step 5: Semantic Matching ===
semantic_matches = []
threshold = 0.85  # You can adjust threshold here

for idx, pred_emb in enumerate(predicted_embeddings):
    sims = cosine_similarity([pred_emb], ground_truth_embeddings)[0]
    max_sim = np.max(sims)
    best_match_idx = np.argmax(sims)

    if max_sim >= threshold:
        semantic_matches.append({
            'Predicted Risk': predicted_texts[idx],
            'Matched Ground Truth Risk': ground_truth_texts[best_match_idx],
            'Similarity Score': round(max_sim, 2)
        })

# === Step 6: Calculate Metrics ===
true_positives = len(semantic_matches)
precision = true_positives / len(predicted_texts) if predicted_texts else 0
recall = true_positives / len(ground_truth_texts) if ground_truth_texts else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

# === Step 7: Print Results ===
print("\n=== Evaluation Metrics (OpenAI Ada-002) ===")
print(f"Threshold: {threshold:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

print("\n=== Semantic Matches Found ===")
for match in semantic_matches:
    print(match)
