import os
import tempfile

from docx import Document
from pypdf import PdfReader


def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def read_docx(uploaded_file):
    document = Document(uploaded_file)
    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def read_cv(uploaded_file):
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    if file_name.endswith(".docx"):
        return read_docx(uploaded_file)
    if file_name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    raise ValueError("Please upload a PDF, DOCX or TXT file.")
