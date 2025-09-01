import pickle
import pandas as pd

# your parquet file
parquet_path = r"C:\Users\SIC\SO-Janitor-Agent\data\processed\top_50_tags_golden_questions.parquet"
# existing id_map
id_map_path = r"C:/Users/SIC/SO-Janitor-Agent/id_map.pkl"
# new id_map to save
new_id_map_path = r"C:/Users/SIC/SO-Janitor-Agent/id_map_fixed.pkl"

# load dataframe
df = pd.read_parquet(parquet_path)

# load old id_map (position → old df.index)
with open(id_map_path, "rb") as f:
    old_id_map = pickle.load(f)

# build new id_map (position → actual Id column)
new_id_map = {pos: df.loc[idx, "Id"] for pos, idx in old_id_map.items()}

# save fixed map
with open(new_id_map_path, "wb") as f:
    pickle.dump(new_id_map, f)

print(f"✅ Fixed id_map saved to {new_id_map_path}")
