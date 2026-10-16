
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

print("Loading dataset...")
df=pd.read_csv("WELFake_ML.csv")

df=df[["content","label"]].dropna()
df["content"]=df["content"].astype(str)
df=df[df["content"].str.strip()!=""]
df["label"]=pd.to_numeric(df["label"],errors="coerce")
df=df.dropna(subset=["label"])
df["label"]=df["label"].astype(int)

print("Dataset shape:",df.shape)
print("Label counts:")
print(df["label"].value_counts())

# Confirm with Member 1 that 0 = Fake and 1 = Real.
X=df["content"]
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

print("Extracting TF-IDF features...")
vectorizer=TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words="english",
    min_df=2
)

X_train_tfidf=vectorizer.fit_transform(X_train)
X_test_tfidf=vectorizer.transform(X_test)

print("Training Multinomial Naive Bayes...")
model=MultinomialNB()
model.fit(X_train_tfidf,y_train)

y_pred=model.predict(X_test_tfidf)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,zero_division=0)
recall=recall_score(y_test,y_pred,zero_division=0)
f1=f1_score(y_test,y_pred,zero_division=0)

print("\nMODEL EVALUATION RESULTS")
print("------------------------")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test,y_pred,zero_division=0))

joblib.dump(model,"model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

pd.DataFrame({
    "actual_label":y_test.to_numpy(),
    "predicted_label":y_pred
}).to_csv("test_predictions.csv",index=False)

metrics={
    "accuracy":float(accuracy),
    "precision":float(precision),
    "recall":float(recall),
    "f1_score":float(f1)
}

with open("metrics.json","w") as f:
    json.dump(metrics,f,indent=4)

print("\nSaved successfully:")
print("model.pkl")
print("vectorizer.pkl")
print("test_predictions.csv")
print("metrics.json")
