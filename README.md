# Smart Resume Analyzer

Smart Resume Analyzer is a **Streamlit-based AI-powered application** that extracts, analyzes, and evaluates resumes to provide insights on candidate qualifications. It utilizes **OCR, NLP, and machine learning techniques** to process PDF resumes and extract relevant details.

## Features
- **Resume Parsing**: Extracts text from PDF resumes using `pdf2image`, `pdfplumber`, and `pytesseract`.
- **Skills & Qualification Analysis**: Uses NLP (`textblob`, `scikit-learn`) to evaluate content.
- **Generative AI Support**: Integrates `google-generativeai` for AI-driven resume enhancement.
- **PDF Output**: Generates structured reports using `fpdf`.
- **Web Interface**: Built with `Streamlit` for an interactive and user-friendly experience.

## Technologies Used
- **Frontend**: Streamlit
- **Backend**: Python
- **OCR & NLP**: `pytesseract`, `pdfplumber`, `scikit-learn`, `textblob`
- **Machine Learning**: `scikit-learn`
- **AI Integration**: Google Generative AI

## Project Structure
```
smart_resume_analyzer/
│── App.py               # Main Streamlit app file
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
│── utils.py             # Helper functions for processing
│── models/              # ML models (if any)
│── data/                # Sample resumes (optional)
│── assets/              # Logos or icons (optional)
```

## Deployment

https://cvboost-7ikp5elntab7sa6s4ugn5h.streamlit.app/


