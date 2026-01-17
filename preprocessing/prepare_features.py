import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------- Paths --------
DATA_PATH = "C:\\Users\\BUNNY YADAV\\RANK-GraphCNN\\data\\merged_raw.csv"
OUTPUT_PATH = "C:\\Users\\BUNNY YADAV\\RANK-GraphCNN\\data\\processed_data.csv"

# -----------------------

GRAPH_ID_COLS = ['srcip', 'dstip', 'sport', 'dsport', 'Stime']
LABEL_COLS = ['Label', 'attack_cat']


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    # ---------------- Drop rows without labels ----------------
    df = df.dropna(subset=['Label'])

    # ---------------- Encode categorical columns ----------------
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = [c for c in cat_cols if c not in GRAPH_ID_COLS + LABEL_COLS]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ---------------- Separate parts ----------------
    labels = df[LABEL_COLS]
    graph_ids = df[GRAPH_ID_COLS]

    features = df.drop(columns=GRAPH_ID_COLS + LABEL_COLS)

    # ---------------- Normalize ----------------
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    features_scaled = pd.DataFrame(
        features_scaled, columns=features.columns
    )

    # ---------------- Combine ----------------
    final_df = pd.concat([graph_ids, features_scaled, labels], axis=1)

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Processed dataset saved to:", OUTPUT_PATH)
    print("Final shape:", final_df.shape)
    print("\nLabel distribution:")
    print(final_df['Label'].value_counts())


if __name__ == "__main__":
    prepare_data()
