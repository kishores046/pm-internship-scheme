# Resume Screening System

A Python-based resume screening tool that uses TF-IDF similarity scoring to match candidate resumes against job descriptions.

## Features
- 📄 PDF text extraction from resumes
- 👤 Automatic candidate name detection
- 🔍 TF-IDF-based similarity scoring
- 📊 CSV-based candidate management
- ⚙️ Configurable allocation threshold

## Requirements
```bash
pip install pandas PyPDF2 scikit-learn
```

## Usage

### As a Library
```python
from resume_matcher import ResumeScreener

# Initialize screener
screener = ResumeScreener(allocation_threshold=0.15)

# Process a resume
result = screener.process_resume('resume.pdf', job_description_text)
print(f"Score: {result['score']:.3f}")

# Update candidates CSV
candidates = screener.update_candidates_csv('candidates.csv', result)
```

### In Google Colab
```python
from resume_matcher import main
main()  # Interactive file upload interface
```

## How It Works
1. Extracts text from PDF resumes
2. Cleans and normalizes text
3. Compares resume against job description using TF-IDF
4. Calculates cosine similarity score (0-1)
5. Allocates candidates above threshold
