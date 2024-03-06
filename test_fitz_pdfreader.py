import fitz  # PyMuPDF for extracting text from PDF

# this function uses fitz module to extract text from pdf
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

print(extract_text_from_pdf('test-resume/Lim Jing Kai Joel (CV).pdf'))