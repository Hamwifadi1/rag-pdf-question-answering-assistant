"""Streamlit entry point for the RAG PDF Question Answering Assistant."""

import streamlit as st

from src.chunker import chunk_pages
from src.embeddings import EmbeddingError, generate_chunk_embeddings
from src.pdf_loader import PDFExtractionError, extract_pdf_pages
from src.vector_store import FAISSVectorStore, VectorStoreError


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
    upload_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("indexed_upload") != upload_signature:
        st.session_state.pop("vector_store", None)
        st.session_state.pop("indexed_upload", None)

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
                "downloads the free model."
            )
            if st.button("Build local FAISS index", type="primary"):
                with st.spinner("Generating embeddings and building FAISS..."):
                    try:
                        embedding_result = generate_chunk_embeddings(chunks)
                        vector_store = FAISSVectorStore.from_embeddings(
                            chunks,
                            embedding_result,
                        )
                    except (EmbeddingError, VectorStoreError) as error:
                        st.error(str(error))
                    else:
                        st.session_state["vector_store"] = vector_store
                        st.session_state["indexed_upload"] = upload_signature
                        st.success("Local embeddings and FAISS index generated successfully.")
                        st.write("Embedding model: all-MiniLM-L6-v2")
                        st.write(
                            "Number of chunks sent for embedding: "
                            f"{embedding_result['chunk_count']}"
                        )
                        st.write(
                            "Embedding vector dimension: "
                            f"{embedding_result['embedding_dimension']}"
                        )
                        st.write(
                            "Number of vectors stored in FAISS: "
                            f"{vector_store.vector_count}"
                        )

            vector_store = st.session_state.get("vector_store")
            if vector_store is not None:
                st.info(f"FAISS index contains {vector_store.vector_count} vector(s).")
                st.subheader("Retrieval test")
                question = st.text_input("Ask a question to retrieve relevant chunks")

                if st.button("Retrieve top 4 chunks"):
                    with st.spinner("Embedding question and searching FAISS..."):
                        try:
                            retrieval = vector_store.retrieve(question, top_k=4)
                        except (EmbeddingError, VectorStoreError) as error:
                            st.error(str(error))
                        else:
                            retrieved_chunks = retrieval["chunks"]
                            st.success(f"Retrieved {len(retrieved_chunks)} chunk(s).")
                            st.write(
                                "Question embedding dimension: "
                                f"{retrieval['question_embedding_dimension']}"
                            )
                            st.write(
                                "Retrieved chunk IDs: "
                                f"{[chunk['chunk_id'] for chunk in retrieved_chunks]}"
                            )
                            st.write(
                                "Retrieved source pages: "
                                f"{[chunk['page_number'] for chunk in retrieved_chunks]}"
                            )

                            for chunk in retrieved_chunks:
                                st.markdown(
                                    f"**Chunk {chunk['chunk_id']} - "
                                    f"Page {chunk['page_number']} - "
                                    f"Similarity {chunk['similarity_score']:.4f}**"
                                )
                                st.text(chunk["text"][:300])
