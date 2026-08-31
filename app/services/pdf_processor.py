import os
import PyPDF2
import docx
from typing import List, Dict, Any
import re

class PDFProcessor:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""
    
    def split_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
        if not text:
            return []
        
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                current_chunk = overlap_text + " " + paragraph
            else:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        if not chunks:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
        text = self.extract_text(file_path)
        if not text:
            return []
        
        chunks = self.split_text(text, chunk_size, chunk_overlap)
        
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                'text': chunk,
                'filename': os.path.basename(file_path),
                'chunk_id': i,
                'source': file_path
            })
        
        return result
