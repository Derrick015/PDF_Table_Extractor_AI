import streamlit as st
import asyncio
import tempfile
import os
import pymupdf
from dotenv import load_dotenv

from modules.pdf_extraction import (
    write_output_final,
    get_validated_table_info,
    get_page_pixel_data,
    process_tables_to_df
)

# Load environment variables from .env
load_dotenv()

def process_single_page(doc, page_no, tmp_file_path, open_api_key, user_text):
    async def _process():
        page = doc.load_page(page_no)
        extracted_text = page.get_text()
        base64_image = get_page_pixel_data(
            pdf_path=tmp_file_path, page_no=page_no, dpi=500, image_type='png'
        )
        num_tables, table_headers, table_location, _ = await get_validated_table_info(
            text_input=extracted_text,
            open_api_key=open_api_key,
            base64_image=base64_image
        )
        if num_tables == 0:
            return None
        return await process_tables_to_df(
            table_headers,
            table_location,
            user_text,
            extracted_text,
            base64_image,
            open_api_key,
            page_no
        )
    return _process()

async def process_all_pages(doc, selected_pages, tmp_file_path, open_api_key, user_text, update_progress):
    tasks = [
        asyncio.create_task(
            process_single_page(doc, page_no, tmp_file_path, open_api_key, user_text)
        )
        for page_no in selected_pages
    ]
    results = []
    total_tasks = len(tasks)
    completed = 0

    for finished_task in asyncio.as_completed(tasks):
        result = await finished_task
        results.append(result)
        completed += 1
        update_progress(completed, total_tasks)

    # Filter out pages that returned None (i.e. no tables found)
    return [res for res in results if res is not None]

def generate_excel_files(output_final, file_name="output"):
    """Generate three Excel files and return them as bytes."""
    os.makedirs("files", exist_ok=True)
    paths = {
        "combined": f"files/{file_name}_page_combined.xlsx",
        "split": f"files/{file_name}_page_split.xlsx",
        "one_sheet": f"files/{file_name}_one_sheet_split.xlsx"
    }

    # Write the final results in each of the three formats
    write_output_final(output_final, excel_path=paths["combined"], option=1)
    write_output_final(output_final, excel_path=paths["split"], option=2)
    write_output_final(output_final, excel_path=paths["one_sheet"], option=3)

    # Read back each file into memory so we can offer them for download
    excel_files = {}
    for key, path in paths.items():
        with open(path, "rb") as f:
            excel_files[key] = f.read()
    return excel_files

def main():
    st.title("Single Slider: Page Range and Dual Preview")

    # 1) Upload PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    user_text = st.text_input("Enter extraction prompt", "Extract all data from the table(s) the header")

    if uploaded_file is not None:
        # Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        st.write(f"Uploaded file saved to temporary path: {tmp_file_path}")

        # Open the PDF
        try:
            doc = pymupdf.open(tmp_file_path)
        except Exception as e:
            st.error(f"Failed to open PDF: {e}")
            return

        total_pages = doc.page_count
        st.write(f"Total pages in document: {total_pages}")

        # 2) Single Slider for start and end page
        st.subheader("Select page range (1-indexed)")
        page_range = st.slider(
            "Drag to pick the start/end pages",
            min_value=1,
            max_value=total_pages,
            value=(1, total_pages)
        )
        start_page, end_page = page_range  # 1-indexed
        selected_pages = range(start_page - 1, end_page)  # convert to 0-indexed

        # 3) Preview both the first and last pages of the range (if distinct)
        st.subheader("Preview of Selected Range")
        col1, col2 = st.columns(2)
        preview_dpi = 300

        # Preview Start Page
        start_page_base64 = get_page_pixel_data(
            pdf_path=tmp_file_path,
            page_no=(start_page - 1),  # 0-index
            dpi=preview_dpi,
            image_type='png'
        )

        with col1:
            st.image("data:image/png;base64," + start_page_base64, caption=f"Start Page: {start_page}")

        # If the range has more than one page, also preview the last page
        if end_page > start_page:
            end_page_base64 = get_page_pixel_data(
                pdf_path=tmp_file_path,
                page_no=(end_page - 1),  # 0-index
                dpi=preview_dpi,
                image_type='png'
            )
            with col2:
                st.image("data:image/png;base64," + end_page_base64, caption=f"End Page: {end_page}")
        else:
            col2.write("Only one page selected, no second preview.")

        # Make sure we have an OpenAI API key
        open_api_key = os.getenv("OPENAI_API_KEY")
        if not open_api_key:
            st.error("OPENAI_API_KEY not found in .env or environment variables.")
            return

        # 4) Button to start processing
        if st.button("Process PDF"):
            if "output_final" not in st.session_state:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(completed, total):
                    progress_bar.progress(completed / total)
                    status_text.text(f"Processed {completed} of {total} pages.")

                # Process asynchronously
                with st.spinner("Processing PDF..."):
                    st.session_state["output_final"] = asyncio.run(
                        process_all_pages(
                            doc,
                            selected_pages,
                            tmp_file_path,
                            open_api_key,
                            user_text,
                            update_progress
                        )
                    )
                st.success("PDF processing complete.")

            # Display the extracted tables
            if st.session_state.get("output_final"):
                st.write("Extracted Tables:")
                for page_data in st.session_state["output_final"]:
                    for df in page_data:
                        st.dataframe(df)

            # Generate Excel files if not already done
            if "excel_files" not in st.session_state:
                with st.spinner("Generating Excel files..."):
                    st.session_state["excel_files"] = generate_excel_files(st.session_state["output_final"])
                st.success("Excel files generated.")

            # Set a flag so the download buttons always show after processing
            st.session_state["processed"] = True

            # Close the PDF document to release the file lock
            doc.close()

            # Remove the temporary file
            try:
                os.remove(tmp_file_path)
            except Exception as e:
                st.error(f"Error removing temporary file: {e}")

        # Outside the button block, always display the download buttons if processing is done
        if st.session_state.get("processed"):
            colA, colB, colC = st.columns(3)
            with colA:
                st.download_button(
                    label="Download Combined (Option 1)",
                    data=st.session_state["excel_files"]["combined"],
                    file_name="output_combined.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with colB:
                st.download_button(
                    label="Download Split (Option 2)",
                    data=st.session_state["excel_files"]["split"],
                    file_name="output_split.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with colC:
                st.download_button(
                    label="One Sheet (Option 3)",
                    data=st.session_state["excel_files"]["one_sheet"],
                    file_name="output_one_sheet.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
