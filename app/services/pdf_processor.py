import PyPDF2
import docx
import os
import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

try:
    from pdf2image import convert_from_path
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

class PDFProcessor:
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean common PDF extraction artifacts - IMPROVED"""
        if not text:
            return text
        
        # Step 1: Replace dots patterns while preserving text structure
        # Match patterns like "text ...................... 7" or "5.1 ......................."
        # Replace with single space
        text = re.sub(r'(?<=[a-zA-Z0-9])\s*\.{4,}\s*(?=[a-zA-Z0-9])', ' ', text)
        
        # Step 2: For dots at end of lines (page numbers, section numbers)
        text = re.sub(r'\.{4,}\s*\d+\s*$', ' ', text, flags=re.MULTILINE)
        
        # Step 3: Remove remaining excessive dots (but leave single dots for decimals, periods)
        text = re.sub(r'\.{3,}', ' ', text)
        
        # Step 4: Remove excessive dashes and underscores
        text = re.sub(r'-{3,}', ' ', text)
        text = re.sub(r'_{3,}', ' ', text)
        
        # Step 5: Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        
        # Step 6: Remove common PDF artifacts
        text = re.sub(r'', '', text)  # Form feed
        text = re.sub(r'\x0c', '', text)  # Another form feed
        
        # Step 7: Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF2 first, fallback to OCR if needed"""
        text = ""
        
        # Try PyPDF2 first (for digital PDFs)
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():  # If we got text
                        # Clean the text
                        page_text = self.clean_extracted_text(page_text)
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
        
        # If PyPDF2 got little/no text, try OCR (for scanned PDFs)
        if len(text.strip()) < 100 and HAS_OCR:  # Less than 100 chars
            print(f"Trying OCR for scanned PDF: {file_path}")
            try:
                ocr_text = self.extract_text_from_pdf_ocr(file_path)
                if ocr_text:
                    # Clean OCR text
                    text = self.clean_extracted_text(ocr_text)
                    print(f"OCR extracted {len(text)} characters")
            except Exception as e:
                print(f"OCR extraction failed: {e}")
        
        if not text.strip():
            raise Exception("Could not extract text from PDF. File may be scanned or corrupted.")
        
        # Final cleaning
        text = self.clean_extracted_text(text)
        return text
    
    def extract_text_from_pdf_ocr(self, file_path: str) -> str:
        """Extract text from scanned PDF using OCR"""
        if not HAS_OCR:
            raise Exception("OCR dependencies not installed. Install: pip install pdf2image pytesseract pillow")
        
        text = ""
        try:
            # Convert PDF to images
            images = convert_from_path(file_path, dpi=300)
            
            # Extract text from each image using OCR
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='eng')
                text += f"Page {i+1}:\n{page_text}\n\n"
            
            return text
        except Exception as e:
            raise Exception(f"OCR extraction error: {str(e)}")
    
    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            text = self.clean_extracted_text(text)
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            text = self.clean_extracted_text(text)
            return text
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    def extract_text(self, file_path: str, file_type: str) -> str:
        if file_type == 'pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_type == 'docx':
            return self.extract_text_from_docx(file_path)
        elif file_type == 'txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def chunk_text(self, text: str) -> List[str]:
        chunks = self.text_splitter.split_text(text)
        # Clean each chunk
        cleaned_chunks = [self.clean_extracted_text(chunk) for chunk in chunks]
        return cleaned_chunks
    
    def process_document(self, file_path: str, file_type: str) -> Dict[str, Any]:
        try:
            text = self.extract_text(file_path, file_type)
            chunks = self.chunk_text(text)
            
            chunks_with_metadata = []
            for i, chunk in enumerate(chunks):
                chunks_with_metadata.append({
                    "text": chunk,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
            
            return {
                "text": text,
                "chunks": chunks_with_metadata,
                "total_chunks": len(chunks),
                "total_characters": len(text)
            }
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
