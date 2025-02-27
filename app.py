# stream lit 
import pymupdf 

import streamlit as st
from modules.pdf_extraction import extract_text_from_pages


# ----------------- Streamlit Front End -----------------

st.title("PDF Text Extractor")

# Upload the PDF file.
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Determine total pages by opening the PDF once.
    pdf_bytes = uploaded_file.getvalue()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    doc.close()
    st.write(f"Total Pages in PDF: **{total_pages}**")

    # Let the user select the extraction option.
    extraction_option = st.selectbox(
        "Select extraction option:",
        ("All pages", "Single page", "Page range", "Multiple pages")
    )

    pages = None  # This will be passed to our function.
    if extraction_option == "Single page":
        page_num = st.number_input("Enter page number", min_value=1, max_value=total_pages, value=1, step=1)
        pages = page_num - 1  # Convert to 0-index
    elif extraction_option == "Page range":
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start page", min_value=1, max_value=total_pages, value=1, step=1)
        with col2:
            end_page = st.number_input("End page (exclusive)", min_value=2, max_value=total_pages + 1, value=total_pages + 1, step=1)
        pages = (start_page - 1, end_page - 1)  # 0-index conversion; end is exclusive
    elif extraction_option == "Multiple pages":
        pages_str = st.text_input("Enter page numbers separated by commas (e.g., 1,3,5)", value="1")
        try:
            pages = [int(p.strip()) - 1 for p in pages_str.split(",") if p.strip()]
        except ValueError:
            st.error("Invalid input for page numbers. Please enter integers separated by commas.")
            pages = None

    if st.button("Extract Text"):
        # The file uploader returns a BytesIO object so we pass it directly.
        extracted_text = extract_text_from_pages(uploaded_file, pages=pages)
        if extracted_text:
            st.text_area("Extracted Text", extracted_text, height=300)
