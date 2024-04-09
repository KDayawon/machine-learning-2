import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import scrolledtext
from tkinter import messagebox

# Load and preprocess the data
data = pd.read_csv('spam.csv')

# Split the data into features and target variable
X = data['Message']
y = data['Category']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=26)

# Convert text data into numerical vectors using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Logistic Regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Function to classify a message
def classify_message():
    message = text_input.get("1.0", 'end-1c')  # Extract text
    if not message.strip():
        messagebox.showerror("Error", "Please enter a message to classify.")
        return
    message_tfidf = tfidf_vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    result_var.set(f"The message is classified as: {prediction.upper()}")

# Creating the GUI
root = Tk()
root.title("Spam or Ham Classifier")
root.geometry("400x300")

text_input = scrolledtext.ScrolledText(root, height=10)
text_input.pack(pady=10)

classify_btn = Button(root, text="Classify Message", command=classify_message)
classify_btn.pack(pady=5)

result_var = StringVar()
result_label = Label(root, textvariable=result_var, font=("Helvetica", 12))
result_label.pack(pady=20)

root.mainloop()
