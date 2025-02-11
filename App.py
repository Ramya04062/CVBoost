import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re
import tempfile

# Load environment variables
load_dotenv()

# Configure Google Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"Direct text extraction failed: {e}")

    print("Falling back to OCR for image-based PDF.")
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
    except Exception as e:
        print(f"OCR failed: {e}")

    return text.strip()

# Function to calculate ATS score using cosine similarity
def calculate_ats_score_gemini(resume_text, job_description):
    prompt = f"""
    Compare the following resume with the given job description and provide an ATS Score (0-100). 
    
    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Give the ATS score based on keyword matching, skill relevance, job role alignment, and overall resume optimization. 
    Respond with only a numerical score out of 100.
    """
    response = model.generate_content(prompt)
    return response.text.strip() if response else "0"


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def generate_cover_letter(resume_text, job_description):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Based on the resume and job description below, generate a professional and personalized cover letter.

        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Provide only the cover letter.
        """

        response = model.generate_content(prompt)  # ✅ FIXED: Removed unnecessary list
        return response.text.strip() if response else "No cover letter generated."

    except Exception as e:
        return f"Error: {str(e)}"




def skill_gap_and_course_suggestions(resume_text, job_description):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        Analyze the job description and resume. Identify the key skills required for the job and compare them with the candidate's resume.
        List the missing skills and suggest relevant courses & certifications.

        Job Description:
        {job_description}

        Resume:
        {resume_text}

        Provide a structured response in the following format:

        ### **Missing Skills**
        - (List missing skills)

        ### **Suggested Courses & Certifications**
        - **Skill Gap**: (Missing skill)
          - **Recommended Course**: (Course Name) - (Platform)
          - **Certification Suggestion**: (Certification Name) - (Issuing Organization)
        """

        response = model.generate_content([prompt])
        return response.text.strip() if response else "No recommendations found."

    except Exception as e:
        return f"Error: {str(e)}"
    

def calculate_resume_strength(resume_text):
    criteria = {
        "Education": ["bachelor", "master", "phd", "degree"],
        "Technical Skills": ["python", "machine learning", "ai", "sql", "cloud"],
        "Experience": ["years", "worked", "internship", "project"],
        "Certifications": ["certified", "course", "completed"]
    }
    
    score = 0
    total_criteria = len(criteria)

    for category, keywords in criteria.items():
        if any(word in resume_text.lower() for word in keywords):
            score += 1

    return round((score / total_criteria) * 100, 2)




# Function to get response from Gemini AI
def analyze_resume(resume_text, job_description=None):
    if not resume_text:
        return {"error": "Resume text is required for analysis."}
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    base_prompt = f"""
    You are an experienced HR with technical expertise in roles such as Data Science, DevOps, AI Engineering, Full Stack Development, etc.
    Review the resume and provide a professional evaluation on the candidate's strengths, weaknesses, skills, and courses for improvement.
    
    Resume:
    {resume_text}
    """
    
    if job_description:
        base_prompt += f"""
        Compare this resume with the job description:
        
        Job Description:
        {job_description}
        """
    
    response = model.generate_content(base_prompt)
    return response.text.strip()

# Streamlit App Configuration
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.markdown("""
    <style>
        .main {background-color: #f4f4f4;}
        .stButton>button {background-color: #0047AB; color: white !important; font-size: 16px; border: none;}
        .stButton>button:active {background-color: #0047AB !important;}
        .stButton>button:hover {color: white !important; background-color: #003580 !important;}
        .stTextArea>textarea {font-size: 14px;}
        .stFileUploader>div>div {border: 2px dashed #0047AB; padding: 15px;}
        div[data-testid="stFileUploader"] {
            max-width: 250px; /* Adjust width */
            margin: auto;
        }
        div[data-testid="stFileUploader"] button {
            font-size: 12px !important; /* Smaller font */
            padding: 4px 8px !important; /* Reduce padding */
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("""
    <h1 style='text-align: center; color: #0047AB;'>CV BOOST</h1>
    <p style='text-align: center;'>Analyze your resume with Google Gemini AI and receive professional insights.</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], help="Supported format: PDF")
with col2:
    job_description = st.text_area("Enter Job Description (Optional)", placeholder="Paste the job description here...")

if uploaded_file is not None:
    st.success("Resume uploaded successfully!")
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_pdf_path = temp_file.name  # Path of the saved file

    resume_text = extract_text_from_pdf("uploaded_resume.pdf")
    
    if st.button("Analyze Resume"):
        with st.spinner("Analyzing resume..."):
            try:
                analysis = analyze_resume(resume_text, job_description)
        
                st.success("Analysis complete!")


                st.markdown("### Analysis Report:")
                st.write(analysis)


                recommendations = skill_gap_and_course_suggestions(resume_text, job_description)

                # Display results
                st.subheader(" Skill Gap Analysis & Course Recommendations")
                st.write(recommendations)
                
                
                
                #Resume Strength
                strength_score = calculate_resume_strength(resume_text)
                st.markdown(f"### Resume Strength: **{strength_score}%**")
                st.progress(int(strength_score))

                # Get ATS Score from Gemini API
                ats_score = calculate_ats_score_gemini(resume_text, job_description)
                st.markdown(f"### ATS Score: **{ats_score}%**")

        # ATS Score Feedback
                if int(ats_score) > 75:
                    st.success("Your resume is highly matched with the job description!")
                elif int(ats_score) > 50:
                    st.warning("Your resume matches moderately. Improve skills & keywords.")
                else:
                    st.error("Low match. Optimize resume with job-relevant keywords.")

                
                cover_letter = generate_cover_letter(resume_text, job_description)
                st.subheader("Generated Cover Letter")
                st.code(cover_letter, language='markdown')
            except Exception as e:
                st.error(f"Analysis failed: {e}")


else:
    st.warning("Please upload a resume in PDF format.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Powered by <b>Streamlit</b> and <b>Google Gemini AI</b></p>
""", unsafe_allow_html=True)
