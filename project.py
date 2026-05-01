print("START")
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
data = pd.read_csv("reddit_depression_dataset.csv", low_memory=False)

# -----------------------------
# STEP 2: Remove unwanted column
# -----------------------------
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# -----------------------------
# STEP 3: Handle missing values
# -----------------------------
data['title'] = data['title'].fillna("")
data['body'] = data['body'].fillna("")

# -----------------------------
# STEP 4: Combine text
# -----------------------------
data['text'] = data['title'] + " " + data['body']

# -----------------------------
# STEP 5: Clean text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data['clean_text'] = data['text'].apply(clean_text)
data = data.sample(5000)



# पहले label को numeric बनाओ (गलत values → NaN)
data['label'] = pd.to_numeric(data['label'], errors='coerce')

# अब NaN वाली rows हटा दो
data = data.dropna(subset=['label'])

# label column check
print("Before cleaning:", data['label'].unique())

# convert to numeric (गलत values → NaN)
data['label'] = pd.to_numeric(data['label'], errors='coerce')

# NaN count check
print("NaN in label:", data['label'].isna().sum())

# remove NaN rows
data = data[data['label'].notna()]

# अब convert करो
data['label'] = data['label'].astype(int)

# final check
print("After cleaning:", data['label'].unique())

# -----------------------------
# STEP 7: Feature Extraction
# -----------------------------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_text']).toarray()
y = data['label']

# -----------------------------
# STEP 8: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 9: Model Training
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# STEP 10: Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# STEP 11: Accuracy
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("Model saved ✅")