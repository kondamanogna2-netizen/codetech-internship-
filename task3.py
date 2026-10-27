import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load Dataset
# Sample dataset - replace with real dataset from Kaggle: Genre Classification Dataset
data = {
    'Plot': [
        "A superhero fights villains and saves the city with explosions and action",
        "Funny friends on a road trip cracking jokes and comedy situations",
        "A couple falls in love in Paris romantic love story",
        "Ghost in haunted house scary horror night with blood",
        "Car chase and gun fights action hero saves world",
        "Standup comedian makes audience laugh hilarious comedy",
        "Boy and girl breakup and patchup romance emotional",
        "Zombie apocalypse terrifying horror survival",
        "Spy mission with fighting and bombs action thriller",
        "Family reunion funny moments comedy drama"
    ],
    'Genre': [
        "Action", "Comedy", "Romance", "Horror", "Action",
        "Comedy", "Romance", "Horror", "Action", "Comedy"
    ]
}

df = pd.DataFrame(data)
# If you have CSV: df = pd.read_csv('movie_genre.csv')

# 2. Fill missing
df['Plot'] = df['Plot'].fillna('')
df['Genre'] = df['Genre'].fillna(df['Genre'].mode()[0])

# 3. Features and Target - NLP Part
X_text = df['Plot']
y = df['Genre']

# 4. TF-IDF Vectorizer - Most Important for this task
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X_text)

# 5. Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 6. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 7. Model - MultinomialNB best for Text Classification
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 9. Save model and vectorizer
with open('genre_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nModel saved!")

# 10. Predict function
def predict_genre(plot_description):
    vec = vectorizer.transform([plot_description])
    pred_encoded = model.predict(vec)
    genre = le.inverse_transform(pred_encoded)
    return genre[0]

# Test Examples
tests = [
    "hero with superpowers fighting enemies",
    "funny joke with friends laughing",
    "ghost in dark house",
    "love letter and romantic dinner"
]

for t in tests:
    print(f"Plot: {t} -> Predicted Genre: {predict_genre(t)}")
