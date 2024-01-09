from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def read_data(file_path, label):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    texts = [line.strip() for line in lines]
    labels = [label] * len(texts)
    return texts, labels


# Read data for Ivan Vazov
vazov_texts, vazov_labels = read_data("vazov.txt", "Vazov")

# Read data for Yordan Yovkov
yovkov_texts, yovkov_labels = read_data("yovkov.txt", "Yovkov")

# Combine the data
texts = vazov_texts + yovkov_texts
labels = vazov_labels + yovkov_labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert text to numerical vectors
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Predict the author of a new text
new_text = "Той не знайше отдих, ни мир, нито сън"
new_text_vectorized = vectorizer.transform([new_text])
predicted_author = classifier.predict(new_text_vectorized)

print(f"Predicted author of the text is: {predicted_author[0]}")
