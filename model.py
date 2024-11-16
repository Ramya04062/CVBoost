import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Sample resumes and labels (1 = good fit, 0 = not a fit)
data = {
    'resume': [
        'Experienced in Python, Data Science, and Machine Learning',
        'Worked with Java, SQL, Databases, and Backend Development',
        'Expert in Machine Learning, AI, and Data Analysis',
        'Skilled in HTML, CSS, JavaScript, and Responsive Design',
        'Strong background in Python, Django, Flask, and Web Development',
        'Experience in React, Node.js, and Frontend technologies'
    ],
    'label': [1, 0, 1, 0, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocess resumes using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['resume'])
y = df['label']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Save the trained model and the TF-IDF vectorizer only if accuracy is satisfactory
if accuracy >= 0.75:  # Threshold can be adjusted as needed
    with open('resume_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    print("Model and vectorizer saved!")
else:
    print("Model not saved due to insufficient accuracy.")

