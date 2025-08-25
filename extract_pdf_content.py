#!/usr/bin/env python3
"""
Script to extract text content from the PDF file
"""
import PyPDF2
import fitz  # PyMuPDF
import os

def extract_text_with_pypdf2(pdf_path):
    """Extract text using PyPDF2"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"PyPDF2 error: {e}")
        return None

def extract_text_with_pymupdf(pdf_path):
    """Extract text using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        print(f"PyMuPDF error: {e}")
        return None

def main():
    pdf_path = "The-Owlgorithms-BPI-DATA-Wave-Initial-Submission.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print("=== Extracting text from PDF ===")
    
    # Try PyMuPDF first (usually better)
    text = extract_text_with_pymupdf(pdf_path)
    if text:
        print("Successfully extracted text with PyMuPDF:")
        print("=" * 50)
        print(text[:2000])  # Print first 2000 characters
        print("=" * 50)
        
        # Save to file
        with open("extracted_pdf_content.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Full content saved to extracted_pdf_content.txt")
        return
    
    # Try PyPDF2 as fallback
    text = extract_text_with_pypdf2(pdf_path)
    if text:
        print("Successfully extracted text with PyPDF2:")
        print("=" * 50)
        print(text[:2000])  # Print first 2000 characters
        print("=" * 50)
        
        # Save to file
        with open("extracted_pdf_content.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Full content saved to extracted_pdf_content.txt")
        return
    
    print("Could not extract text from PDF. The file might be image-based or encrypted.")

if __name__ == "__main__":
    main() 