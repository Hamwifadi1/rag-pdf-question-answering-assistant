"""Streamlit entry point for the RAG PDF Question Answering Assistant."""

import streamlit as st

from src.chunker import chunk_pages
from src.embeddings import EmbeddingError, generate_chunk_embeddings
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

        chunks = chunk_pages(pages)
        st.info(f"Generated {len(chunks)} chunk(s).")
        st.caption(f"Debug: number of generated chunks = {len(chunks)}")

        st.subheader("Chunk preview")
        if not chunks:
            st.warning("No chunks were generated because the PDF has no extractable text.")
        else:
            preview_chunks = chunks[:3]
            for chunk in preview_chunks:
                st.markdown(
                    f"**Chunk {chunk['chunk_id']} - Page {chunk['page_number']}**"
                )
                st.text(chunk["text"][:300])

            if len(chunks) > len(preview_chunks):
                st.caption("Preview limited to the first 3 chunks and 300 characters each.")

            st.subheader("Embedding test")
            st.caption(
                "Embeddings run locally with all-MiniLM-L6-v2. The first run "
                "downloads the free model; FAISS storage is not implemented yet."
            )
            if st.button("Generate local embeddings", type="primary"):
                with st.spinner("Loading the local model and generating embeddings..."):
                    try:
                        embedding_result = generate_chunk_embeddings(chunks)
                    except EmbeddingError as error:
                        st.error(str(error))
                    else:
                        st.success("Local embeddings generated successfully.")
                        st.write("Embedding model: all-MiniLM-L6-v2")
                        st.write(
                            "Number of chunks sent for embedding: "
                            f"{embedding_result['chunk_count']}"
                        )
                        st.write(
                            "Embedding vector dimension: "
                            f"{embedding_result['embedding_dimension']}"
                        )
