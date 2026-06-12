"""Streamlit entry point for the RAG PDF Question Answering Assistant."""

import streamlit as st


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
    help="PDF processing and question answering will be added in later milestones.",
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")
    st.info("PDF text extraction is not implemented yet. That comes in Milestone 2.")

