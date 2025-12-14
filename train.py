import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import extract_features

ai_df = pd.read_csv("data/ai_texts.csv")
human_df = pd.read_csv("data/human_texts.csv")

df = pd.concat([ai_df, human_df], ignore_index=True)

X = df["text"].apply(extract_features).tolist()
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
