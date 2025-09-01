import re
import pickle
import argparse
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def clean_text(t: str) -> str:
    """Remove HTML tags and extra whitespace."""
    if not isinstance(t, str):
        t = str(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def build_index(
    parquet_path: str,
    out_index: str,
    out_map: str,
    model_name: str,
    batch: int,
    text_cols: list[str]
):
    # Load dataset
    df = pd.read_parquet(parquet_path)

    # Apply clean_text to each cell in the text columns
    texts = (
        df[text_cols[0]].astype(str).apply(clean_text) + " " +
        df[text_cols[1]].astype(str).apply(clean_text)
    ).tolist()

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Create embeddings in batches
    embeddings = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i + batch]
        batch_emb = model.encode(batch_texts, show_progress_bar=True)
        embeddings.append(batch_emb)

    embeddings = np.vstack(embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, out_index)

    # Save ID map
    id_map = {i: idx for i, idx in enumerate(df.index.tolist())}
    with open(out_map, "wb") as f:
        pickle.dump(id_map, f)

    print(f"✅ Index saved to {out_index}")
    print(f"✅ ID map saved to {out_map}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from Parquet file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--out_index", type=str, required=True, help="Path to save FAISS index")
    parser.add_argument("--out_map", type=str, required=True, help="Path to save ID map (pickle)")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model name")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for embeddings")
    parser.add_argument("--text_cols", nargs=2, required=True,
                        help="Two column names from the dataframe to combine as text")

    args = parser.parse_args()

    build_index(
        parquet_path=args.input,
        out_index=args.out_index,
        out_map=args.out_map,
        model_name=args.model,
        batch=args.batch,
        text_cols=args.text_cols,
    )
