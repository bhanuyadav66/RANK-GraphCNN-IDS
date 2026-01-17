import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/processed_data.csv"


def train_rf():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=['Label', 'attack_cat', 'srcip', 'dstip', 'sport', 'dsport', 'Stime'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nRandom Forest Baseline Results")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    train_rf()
