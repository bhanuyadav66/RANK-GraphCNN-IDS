import pandas as pd

# -------- Paths --------
ATTACK_PATH = "C:\\Users\\BUNNY YADAV\\RANK-GraphCNN\\data\\darpa.csv"
NORMAL_PATH = "C:\\Users\\BUNNY YADAV\\RANK-GraphCNN\\data\\testData.csv"
OUTPUT_PATH = "data/merged_raw.csv"

# -----------------------

def merge_and_label():
    # Load datasets
    attack_df = pd.read_csv(ATTACK_PATH)
    normal_df = pd.read_csv(NORMAL_PATH)

    # ---------------- Label ATTACK data ----------------
    # dapra.csv already has Label & attack_cat
    attack_df['Label'] = 1
    if 'attack_cat' not in attack_df.columns:
        attack_df['attack_cat'] = 'Attack'

    # ---------------- Label NORMAL data ----------------
    normal_df['Label'] = 0
    normal_df['attack_cat'] = 'Normal'

    # ---------------- Align columns ----------------
    # Ensure both have same columns
    for col in attack_df.columns:
        if col not in normal_df.columns:
            normal_df[col] = None

    for col in normal_df.columns:
        if col not in attack_df.columns:
            attack_df[col] = None

    # Reorder columns to match
    normal_df = normal_df[attack_df.columns]

    # ---------------- Merge ----------------
    merged_df = pd.concat([attack_df, normal_df], ignore_index=True)

    # ---------------- Save ----------------
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print("Merged dataset saved to:", OUTPUT_PATH)
    print("Final shape:", merged_df.shape)
    print("\nLabel distribution:")
    print(merged_df['Label'].value_counts())


if __name__ == "__main__":
    merge_and_label()
