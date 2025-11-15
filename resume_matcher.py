"""
Resume Screening and Candidate Allocation System
Matches candidate resumes against job descriptions using TF-IDF similarity scoring.
"""

import os
import pandas as pd
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeScreener:
    """Handles resume screening and candidate allocation"""
    
    def __init__(self, allocation_threshold=0.1):
        """
        Initialize the resume screener.
        
        Args:
            allocation_threshold: Minimum score for candidate allocation (default: 0.1)
        """
        self.allocation_threshold = allocation_threshold
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text += extracted_page_text
        except Exception as e:
            print(f"Warning: Failed to read {pdf_path}: {e}")
        return text
    
    @staticmethod
    def clean_text(text):
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned and normalized text
        """
        text = text or ""
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()
    
    @staticmethod
    def extract_name_from_resume(text):
        """
        Extract candidate name from resume text.
        Looks for name patterns in the first few lines.
        
        Args:
            text: Resume text
            
        Returns:
            Extracted name or None
        """
        lines = text.split('\n')[:10]
        
        for line in lines:
            line = line.strip()
            if len(line) > 3 and len(line) < 50:
                # Skip common resume headers
                skip_words = ['resume', 'cv', 'curriculum', 'vitae', 'profile', 
                             'objective', 'summary', 'contact', 'email', 'phone', 'address']
                if any(word in line.lower() for word in skip_words):
                    continue
                
                # Look for name patterns
                if re.match(r'^[A-Za-z\s\.]+$', line) and len(line.split()) >= 2:
                    return line.title()
        
        return None
    
    def calculate_similarity_score(self, resume_text, job_description):
        """
        Calculate similarity score between resume and job description.
        
        Args:
            resume_text: Cleaned resume text
            job_description: Cleaned job description text
            
        Returns:
            Similarity score (0-1)
        """
        corpus = [resume_text, job_description]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])
    
    def process_resume(self, resume_path, job_description, resume_id=None):
        """
        Process a single resume and calculate match score.
        
        Args:
            resume_path: Path to resume PDF
            job_description: Job description text
            resume_id: Optional resume ID (defaults to filename without extension)
            
        Returns:
            Dictionary with resume_id, name, score, and allocated status
        """
        if resume_id is None:
            resume_id = os.path.splitext(os.path.basename(resume_path))[0]
        
        # Extract and clean resume text
        raw_text = self.extract_text_from_pdf(resume_path)
        clean_resume = self.clean_text(raw_text)
        
        # Extract candidate name
        candidate_name = self.extract_name_from_resume(raw_text)
        if not candidate_name:
            candidate_name = resume_id
        
        # Calculate similarity score
        score = self.calculate_similarity_score(clean_resume, job_description)
        
        # Determine allocation
        allocated = score > self.allocation_threshold
        
        return {
            'id': resume_id,
            'name': candidate_name,
            'score': score,
            'allocated': allocated
        }
    
    def update_candidates_csv(self, candidates_csv, result):
        """
        Update candidates CSV with screening results.
        
        Args:
            candidates_csv: Path to candidates CSV file
            result: Dictionary with screening results
            
        Returns:
            Updated DataFrame
        """
        # Load candidates
        candidates = pd.read_csv(candidates_csv, dtype=str)
        
        if 'id' not in candidates.columns:
            raise ValueError("CSV must contain 'id' column")
        
        candidates['id'] = candidates['id'].astype(str)
        candidates = candidates.set_index('id')
        
        # Initialize columns if they don't exist
        if 'score' not in candidates.columns:
            candidates['score'] = pd.NA
        if 'name' not in candidates.columns:
            candidates['name'] = ''
        if 'allocated' not in candidates.columns:
            candidates['allocated'] = False
        
        resume_id = result['id']
        
        if resume_id in candidates.index:
            # Update existing candidate
            candidates.loc[resume_id, 'score'] = result['score']
            candidates.loc[resume_id, 'allocated'] = result['allocated']
            
            # Update name only if empty
            if pd.isna(candidates.loc[resume_id, 'name']) or candidates.loc[resume_id, 'name'] == '':
                candidates.loc[resume_id, 'name'] = result['name']
        else:
            # Add new candidate
            new_row = {
                'name': result['name'],
                'score': result['score'],
                'allocated': result['allocated']
            }
            
            # Add default values for other columns
            for col in candidates.columns:
                if col not in ['name', 'score', 'allocated']:
                    new_row[col] = ''
            
            candidates.loc[resume_id] = new_row
        
        return candidates.reset_index()


def main():
    """Main execution function for Google Colab"""
    from google.colab import files
    
    # Upload files
    print("📂 Upload your resume PDF")
    uploaded = files.upload()
    resume_file = list(uploaded.keys())[0]
    
    print("📂 Upload candidates CSV")
    uploaded_csv = files.upload()
    candidates_csv = list(uploaded_csv.keys())[0]
    
    print("📂 Upload job description (optional)")
    uploaded_jd = files.upload()
    jobdesc_file = list(uploaded_jd.keys())[0] if uploaded_jd else None
    
    # Load job description
    if jobdesc_file:
        if jobdesc_file.lower().endswith('.pdf'):
            job_desc_text = ResumeScreener.extract_text_from_pdf(jobdesc_file)
        else:
            with open(jobdesc_file, 'r', encoding='utf-8') as f:
                job_desc_text = f.read()
        job_desc_text = ResumeScreener.clean_text(job_desc_text)
    else:
        # Default job description
        job_desc_text = """
        We are looking for a skilled Python Developer with experience in:
        - Python programming
        - Data analysis and Pandas
        - Machine learning basics
        - Web development (Django/Flask)
        - Problem-solving and algorithms
        """
        job_desc_text = ResumeScreener.clean_text(job_desc_text)
    
    # Initialize screener
    screener = ResumeScreener(allocation_threshold=0.1)
    
    # Process resume
    result = screener.process_resume(resume_file, job_desc_text)
    
    # Update CSV
    candidates = screener.update_candidates_csv(candidates_csv, result)
    
    # Save results
    output_csv = "allocations.csv"
    candidates.to_csv(output_csv, index=False)
    
    # Display results
    print(f"\n✅ Screening complete!")
    print(f"📊 Candidate: {result['name']}")
    print(f"📊 Score: {result['score']:.3f}")
    print(f"📊 Allocated: {'Yes' if result['allocated'] else 'No'}")
    print(f"\n💾 Results saved to {output_csv}")
    
    # Download results
    files.download(output_csv)


if __name__ == "__main__":
    main()
