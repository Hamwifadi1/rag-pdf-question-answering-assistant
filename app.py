"""Streamlit entry point for the RAG PDF Question Answering Assistant."""

import streamlit as st

from src.pdf_loader import PDFExtractionError, extract_pdf_pages


st.set_page_config(
    page_title="RAG PDF Question Answering Assistant",
    page_icon="PDF",
    layout="centered",
)

st.title("RAG PDF Question Answering Assistant")
st.write("Upload one PDF file to get started.")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False,
    help="Upload one PDF to extract and preview its text.",
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    try:
        pages = extract_pdf_pages(uploaded_file)
    except PDFExtractionError as error:
        st.error(str(error))
    else:
        st.info(f"Extracted {len(pages)} page(s).")
        st.caption(f"Debug: number of extracted pages = {len(pages)}")

        st.subheader("Text preview")
        preview_pages = pages[:3]
        for page in preview_pages:
            preview = page["text"][:500].strip()
            st.markdown(f"**Page {page['page_number']}**")
            st.text(preview or "No extractable text found on this page.")

        if len(pages) > len(preview_pages):
            st.caption("Preview limited to the first 3 pages and 500 characters per page.")
