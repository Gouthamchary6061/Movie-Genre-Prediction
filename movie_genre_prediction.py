# =====================================
# Project: Movie Genre Prediction
# Technologies: Python, Pandas, NLTK, Scikit-learn
# =====================================

# Step 1: Import Libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
nltk.download('wordnet')

# =====================================
# Step 2: Load Dataset
# Replace 'movies_dataset.csv' with your dataset path
# CSV columns: plot_summary, genre
# =====================================
df = pd.read_csv('movies_dataset.csv')
print(df.head())

# =====================================
# Step 3: Text Preprocessing
# =====================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['Cleaned_Plot'] = df['plot_summary'].apply(clean_text)

# =====================================
# Step 4: Feature Extraction
# =====================================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Plot'])
y = df['genre']  # Single-label genre

# =====================================
# Step 5: Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================
# Step 6: Train Classifier
# =====================================
model = MultinomialNB()
model.fit(X_train, y_train)

# =====================================
# Step 7: Evaluate Model
# =====================================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =====================================
# Step 8: Predict New Movie Genres
# =====================================
new_movies = [
    "A young man discovers his magical powers while attending a school for wizards.",
    "A detective investigates a series of mysterious murders in the city.",
    "Two friends embark on a hilarious road trip across the country."
]

new_cleaned = [clean_text(plot) for plot in new_movies]
X_new = vectorizer.transform(new_cleaned)
predictions = model.predict(X_new)

for plot, pred in zip(new_movies, predictions):
    print(f"Plot: {plot}\nPredicted Genre: {pred}\n")
