import os
import pdfplumber
import PyPDF2
from werkzeug.utils import secure_filename

def allowed_file(filename, allowed_extensions):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_pdf(file, upload_folder):
    """Save the uploaded PDF file to disk"""
    if file and allowed_file(file.filename, {'pdf'}):
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        return file_path
    return None

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    text = ""
    
    try:
        # Using pdfplumber for better text extraction with layout preservation
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                    
        # If pdfplumber didn't extract anything useful, try PyPDF2 as fallback
        if not text.strip():
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
    
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        text = ""
        
    return text.strip() 