import pandas as pd

DATA_PATH = "C:\\Users\\BUNNY YADAV\\RANK-GraphCNN\\data\\darpa.csv"

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_info(df):
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)
    print("\nLabel distribution:")
    print(df['Label'].value_counts())
    print("\nAttack categories:")
    print(df['attack_cat'].value_counts())

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    basic_info(df)
