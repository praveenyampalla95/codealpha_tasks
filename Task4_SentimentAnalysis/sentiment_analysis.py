import pandas as pd

df = pd.read_csv("amazon_reviews.csv")

print(df.head())

print(df.columns)

print(df.info())
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["Review"])

y = df["Label"]

print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
from sklearn.naive_bayes import MultinomialNB

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")
pred = model.predict(X_test)

print(pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)
from sklearn.metrics import classification_report

print("\nClassification Report")

print(classification_report(y_test, pred))
reviews = [
    "This product is amazing",
    "Worst product ever",
    "I love this phone",
    "I hate this item"
]

for review in reviews:

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    if prediction[0] == 1:
        print(review, "-> Positive Review")
    else:
        print(review, "-> Negative Review")

    

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix")

print(cm)
small_df = df.sample(n=10000, random_state=42)

small_df.to_csv("amazon_reviews_sample.csv", index=False)

print("Sample dataset created successfully!")