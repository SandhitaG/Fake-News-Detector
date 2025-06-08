import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
real = pd.read_csv("C:\\Users\\user\\Desktop\\Fake News Detector\\App\\True.csv")
fake = pd.read_csv("C:\\Users\\user\\Desktop\\Fake News Detector\\App\\Fake.csv")


# Add labels: 1 = Real, 0 = Fake
real['label'] = 1
fake['label'] = 0

# Combine datasets
data = pd.concat([real, fake], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

# Features and labels
X = data['text']
y = data['label']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Ensure 'App' directory exists
os.makedirs("App", exist_ok=True)

# Save model and vectorizer to App/ folder
with open("C:\\Users\\user\\Desktop\\Fake News Detector\\App\\model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("C:\\Users\\user\\Desktop\\Fake News Detector\\App\\vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


print("✅ Model and vectorizer saved to App/")
