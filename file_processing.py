import fitz  # PyMuPDF

def process_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "text/plain":
        return extract_text_from_txt(file)
    else:
        raise ValueError("Unsupported file type")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_txt(file):
    content = file.read()
    return content.decode("utf-8")
