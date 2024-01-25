import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def read_data(file_path, label):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    texts = [line.strip() for line in lines]
    labels = [label] * len(texts)
    return texts, labels


def predict_author():
    # Read input values from the form
    text_to_determine = text_input.get("1.0", "end-1c")
    test_size_value = float(test_size_input.get())
    random_state_value = int(random_state_input.get())

    # Read data for Ivan Vazov
    vazov_texts, vazov_labels = read_data("vazov.txt", "Vazov")

    # Read data for Yordan Yovkov
    yovkov_texts, yovkov_labels = read_data("yovkov.txt", "Yovkov")

    # Combine the data
    texts = vazov_texts + yovkov_texts
    labels = vazov_labels + yovkov_labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size_value,
                                                        random_state=random_state_value)

    # Convert text to numerical vectors
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Create and train the Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    # Predict the author of the new text
    new_text_vectorized = vectorizer.transform([text_to_determine])
    predicted_author = classifier.predict(new_text_vectorized)

    # Display the predicted author
    predicted_author_textbox.delete(1.0, tk.END)
    predicted_author_textbox.insert(tk.END, predicted_author[0])


# Create the main application window
root = tk.Tk()
root.title("Author Prediction")

# Create input fields and labels
text_label = tk.Label(root, text="Enter text to determine author:")
text_label.pack()
text_input = scrolledtext.ScrolledText(root, height=4, width=50)
text_input.pack()

test_size_label = tk.Label(root, text="Test Size:")
test_size_label.pack()
test_size_input = tk.Entry(root)
test_size_input.insert(tk.END, "0.2")  # Default value
test_size_input.pack()

random_state_label = tk.Label(root, text="Random State:")
random_state_label.pack()
random_state_input = tk.Entry(root)
random_state_input.insert(tk.END, "42")  # Default value
random_state_input.pack()

# Create a button to trigger the prediction
predict_button = tk.Button(root, text="Predict Author", command=predict_author)
predict_button.pack()

# Create a textbox to display the predicted author
predicted_author_textbox = tk.Text(root, height=1, width=50)
predicted_author_textbox.pack()

# Run the application
root.mainloop()
