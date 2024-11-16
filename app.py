import os
import pickle
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
with open('resume_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html', prediction=None)

# Route to handle file upload and analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the 'resume' field is in the request
        if 'resume' not in request.files:
            return render_template('index.html', prediction='No resume file uploaded')

        uploaded_file = request.files['resume']
        
        # Check if the uploaded file is valid
        if uploaded_file.filename == '':
            return render_template('index.html', prediction='No file selected')

        # Check for the 'role' input
        role = request.form.get('role')  # Use get to avoid KeyError

        if not role:
            return render_template('index.html', prediction='Role input is required')

        # Save the uploaded file in the 'resumes' folder
        file_path = os.path.join('resumes', uploaded_file.filename)
        uploaded_file.save(file_path)

        # Read the resume text
        with open(file_path, 'r') as file:
            resume_text = file.read()
        
        # Transform the resume text using the vectorizer
        resume_vector = tfidf.transform([resume_text])
        
        # Make a prediction using the trained model
        prediction = model.predict(resume_vector)

        # Determine result
        result = 'Good Fit' if prediction[0] == 1 else 'Not a Fit'
        return render_template('index.html', prediction=f'Resume uploaded successfully! Role: {role}, Prediction: {result}')
    
    return render_template('index.html', prediction='Failed to upload resume')

if __name__ == '__main__':
    if not os.path.exists('resumes'):
        os.makedirs('resumes')
    app.run(debug=True)




