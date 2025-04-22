import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_csv('combined_sarcasm_dataset.csv')
data = data.dropna(subset=['Content'])
df = pd.DataFrame({
    'text': data['Content'],
    'label': data['Label']
})

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(max_features=10000, lowercase=True, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
predictions = lr_model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"F1 Score: {f1_score(y_test, predictions):.4f}")
