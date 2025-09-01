import argparse
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_index(index_path, map_path):
    index = faiss.read_index(index_path)
    with open(map_path, "rb") as f:
        id_map = pickle.load(f)
    return index, id_map

def main():
    parser = argparse.ArgumentParser(description="Search FAISS index for similar questions.")
    parser.add_argument("--index", required=True, help="Path to FAISS index file")
    parser.add_argument("--map", required=True, help="Path to ID map pickle file")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--data", default="data/processed/top_50_tags_golden_questions.parquet",
                        help="Path to the parquet file with questions data")
    args = parser.parse_args()

    # Load FAISS index and id_map
    index, id_map = load_index(args.index, args.map)

    # Load dataset (to fetch titles)
    df = pd.read_parquet(args.data)

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode query
    query_vec = model.encode([args.query])
    query_vec = query_vec.astype("float32")

    # Search
    D, I = index.search(query_vec, k=5)  # top 5 results

    print(f"\n🔎 Query: {args.query}\n")
    print("Top Results:")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        qid = id_map[idx]
        row = df[df["Id"] == qid]
        if row.empty:
            print(f"{rank}. ❌ Question with Id={qid} not found in dataset [Similarity: {score:.2f}]")
        else:
            row = row.iloc[0]
            print(f"{rank}. {row['Title']}  (Tags: {row['Tags']}, Score: {row['Score']}) [Similarity: {score:.2f}]")

if __name__ == "__main__":
    main()
