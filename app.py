import streamlit as st
import os
import tempfile
import time
import logging
import asyncio
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
import zipfile
import io

from modules.pdf_extraction import (
    get_page_pixel_data,
    get_validated_table_info,
    process_tables_to_df,
    write_output_final
)

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure logging
log_file = os.path.join("logs", "app.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will continue to show logs in console
    ]
)

# Log the start of the application
logging.info("Starting PDF Table Extractor AI application")

# Page configuration
st.set_page_config(
    page_title="PDF Table Extractor AI",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'output_final' not in st.session_state:
    st.session_state.output_final = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = "output_file"
if 'custom_pages_last' not in st.session_state:
    st.session_state.custom_pages_last = ""

# Load environment variables
load_dotenv()
open_api_key = os.getenv('OPENAI_API_KEY')
if not open_api_key:
    st.error("OPENAI_API_KEY is not set. Please check your .env file.")
    st.stop()

# Initialize OpenAI client
openai_client = OpenAI(api_key=open_api_key)

# App title and description
st.title("PDF Table Extractor AI")
st.markdown("Upload a PDF file to extract tables.")

# Sidebar for options
with st.sidebar:
    # st.header("Settings")
    
    # Create a files directory if it doesn't exist
    if not os.path.exists("files"):
        os.makedirs("files")
        
    # Output file name
    file_name = st.text_input("File name", value=st.session_state.file_name)
    st.session_state.file_name = file_name
    
    # Use text_area instead of text_input for more space
    user_text = st.text_area(
        "Instructions for AI", 
        value="Extract all data from the table(s)",
        height=200  # Make the box much taller
    )
    
    # Validate inputs - ensure defaults if empty
    if not file_name.strip():
        file_name = "output_file"
        st.session_state.file_name = file_name
        st.warning("Using default filename 'output_file' as none was provided.")
    
    if not user_text.strip():
        user_text = "Extract all data from the table(s)"
        st.warning("Using default instructions 'Extract all data from the table(s)' as none were provided.")

# File upload section
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

# Main processing logic
if uploaded_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    # Load PDF and show preview
    try:
        doc = pymupdf.open(pdf_path)
        total_pages = doc.page_count
        
        st.success(f"Successfully loaded PDF with {total_pages} pages.")
        
        # Page range selection
        st.subheader("Page Range Selection")
        range_option = st.radio("Select pages to process:", 
                               ["All pages", "Specific range", "Custom pages"])
        
        if range_option == "All pages":
            page_indices = list(range(total_pages))
            st.info(f"Processing all {total_pages} pages")
            
        elif range_option == "Specific range":
            col1, col2 = st.columns(2)
            with col1:
                start_page = st.number_input("Start page", min_value=1, max_value=total_pages, value=1)
            with col2:
                end_page = st.number_input("End page", min_value=start_page, max_value=total_pages, value=min(start_page + 4, total_pages))
            
            page_indices = list(range(start_page - 1, end_page))
            st.info(f"Processing pages {start_page} to {end_page} (total: {len(page_indices)} pages)")
            
            # Display start and end page previews
            st.subheader("Range Preview")
            preview_col1, preview_col2 = st.columns(2)
            
            with preview_col1:
                st.markdown(f"**Start Page ({start_page})**")
                start_page_index = start_page - 1
                page = doc.load_page(start_page_index)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption=f"Page {start_page}", use_container_width=True)
            
            with preview_col2:
                st.markdown(f"**End Page ({end_page})**")
                end_page_index = end_page - 1
                page = doc.load_page(end_page_index)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption=f"Page {end_page}", use_container_width=True)
            
        else:  # Custom pages
            custom_pages = st.text_input("Enter page numbers separated by commas (e.g., 1,3,5,8)")
            preview_button = st.button("Preview Pages")
            
            if custom_pages and (preview_button or 'custom_pages_last' in st.session_state and st.session_state.custom_pages_last == custom_pages):
                try:
                    # Store current custom pages value in session state to maintain preview between interactions
                    st.session_state.custom_pages_last = custom_pages
                    
                    page_nums = [int(p.strip()) for p in custom_pages.split(",")]
                    # Validate page numbers
                    valid_pages = [p for p in page_nums if 1 <= p <= total_pages]
                    page_indices = [p - 1 for p in valid_pages]  # Convert to 0-based indices
                    
                    if len(valid_pages) != len(page_nums):
                        st.warning(f"Some page numbers were out of range and will be ignored. Valid range: 1-{total_pages}")
                    
                    st.info(f"Processing {len(page_indices)} pages: {', '.join(map(str, valid_pages))}")
                    
                    # Display previews of custom pages (up to 4)
                    if valid_pages:
                        st.subheader("Page Previews")
                        preview_pages = valid_pages[:4]  # Show max 4 previews
                        
                        columns = st.columns(min(len(preview_pages), 4))
                        for i, page_num in enumerate(preview_pages):
                            with columns[i]:
                                st.markdown(f"**Page {page_num}**")
                                page = doc.load_page(page_num - 1)
                                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                                img_bytes = pix.tobytes("png")
                                st.image(img_bytes, caption=f"Page {page_num}", use_container_width=True)
                        
                        if len(valid_pages) > 4:
                            st.info(f"Showing first 4 of {len(valid_pages)} selected pages")
                except ValueError:
                    st.error("Please enter valid page numbers separated by commas")
                    page_indices = []
            else:
                # Initialize session state if first time
                if 'custom_pages_last' not in st.session_state:
                    st.session_state.custom_pages_last = ""
                    
                page_indices = []
                if not custom_pages:
                    st.warning("Please specify at least one page number")
                elif not preview_button:
                    st.info("Click 'Preview Pages' to see the selected pages")
        
        # Show the process button only if page_indices is not empty
        if page_indices:
            # Only show the process button if we haven't completed processing or if we're reprocessing
            process_button = st.button("Process Selected Pages")
            
            if process_button:
                # Reset the processing state
                st.session_state.processing_complete = False
                st.session_state.output_final = []
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                async def process_pages():
                    tasks = []
                    results_output = []
                    
                    try:
                        status_text.text("Initializing page processing...")
                        
                        async with asyncio.TaskGroup() as tg:
                            for i, page_no in enumerate(page_indices):
                                status_text.text(f"Processing page {page_no + 1}...")
                                progress_bar.progress((i / len(page_indices)) * 0.5)  # Update to 50% through the process
                                
                                page = doc.load_page(page_no)
                                
                                tabs = page.find_tables()
                                num_tables_0 = len(tabs.tables)
                                
                                # Check for the presence of tables with pymupdf. This will mean images with tables will be ignored. 
                                if num_tables_0 == 0:
                                    st.info(f"No tables found on page {page_no + 1}")
                                    continue
                                
                                extracted_text = page.get_text()
                                
                                base64_image = get_page_pixel_data(
                                    pdf_path=pdf_path,
                                    page_no=page_no,
                                    dpi=500,
                                    image_type='png'
                                )
                                
                                num_tables, table_headers, confidence_score = await get_validated_table_info(
                                    text_input=extracted_text,
                                    user_text=user_text,
                                    open_api_key=open_api_key,
                                    base64_image=base64_image
                                )
                                
                                # Check for the presence of tables with LLM. 
                                if num_tables == 0:
                                    st.info(f"No tables found on page {page_no + 1}")
                                    continue
                                
                                
                                tasks.append(tg.create_task(process_tables_to_df(
                                    table_headers,
                                    user_text,
                                    extracted_text,
                                    base64_image,
                                    open_api_key,
                                    page_no
                                )))
                            
                            # Await all tasks to complete
                            for j, task in enumerate(tasks):
                                results_output.append(await task)
                                progress_bar.progress(0.5 + ((j + 1) / len(tasks)) * 0.5)  # Update from 50% to 100%
                        
                        status_text.text("Processing complete!")
                        return results_output
                        
                    except Exception as e:
                        st.error("An issue occurred during processing. Please try again. If the issue persists, try with a different page range or check your PDF file.")
                        logging.error(f"Processing error details: {str(e)}")
                        return []
                
                # Start processing
                start_time = time.time()
                with st.spinner("Processing PDF tables..."):
                    output_final = asyncio.run(process_pages())
                    # Store the output in session state
                    st.session_state.output_final = output_final
                    st.session_state.processing_complete = True
                
                # Calculate elapsed time
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.success(f"Processing completed in {elapsed_time:.2f} seconds")
            
            # Check if processing has been completed (either in this run or a previous one)
            if st.session_state.processing_complete:
                output_final = st.session_state.output_final
                
                # Only show download options if we have results
                if output_final and len(output_final) > 0:
                    st.subheader("Download Options")
                    
                    # Create columns for download buttons
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Make sure the files directory exists
                    if not os.path.exists("files"):
                        os.makedirs("files")
                    
                    # Save results and provide download buttons
                    with col1:
                        combined_file = f'files/{file_name}_concatenated.xlsx'
                        write_output_final(output_final, excel_path=combined_file, option=1)
                        
                        with open(combined_file, "rb") as file:
                            st.download_button(
                                label="Download Format 1",
                                data=file,
                                file_name=f"{file_name}_concatenated.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        st.caption("All tables concatenated on a single sheet.")
                    
                    with col2:
                        split_file = f'files/{file_name}_page_per_sheet.xlsx'
                        write_output_final(output_final, excel_path=split_file, option=2)
                        
                        with open(split_file, "rb") as file:
                            st.download_button(
                                label="Download Format 2",
                                data=file,
                                file_name=f"{file_name}_page_per_sheet.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        st.caption("All tables on a page per sheet")
                    
                    with col3:
                        one_sheet_file = f'files/{file_name}_all_tables_on_one_sheet.xlsx'
                        write_output_final(output_final, excel_path=one_sheet_file, option=3)
                        
                        with open(one_sheet_file, "rb") as file:
                            st.download_button(
                                label="Download Format 3",
                                data=file,
                                file_name=f"{file_name}_all_tables_on_one_sheet.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        st.caption("All tables on one sheet")
                    
                    with col4:
                        # Create a zip file containing all three Excel files
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Add each Excel file to the zip
                            for file_path, file_type in [
                                (combined_file, "concatenated"),
                                (split_file, "page_per_sheet"),
                                (one_sheet_file, "all_tables_on_one_sheet")
                            ]:
                                with open(file_path, "rb") as f:
                                    zip_file.writestr(f"{file_name}_{file_type}.xlsx", f.read())
                        
                        # Set buffer position to start
                        zip_buffer.seek(0)
                        
                        st.download_button(
                            label="Download All Formats (ZIP)",
                            data=zip_buffer,
                            file_name=f"{file_name}_all_formats.zip",
                            mime="application/zip"
                        )
                        st.caption("All formats in a single ZIP file")
                    
                else:
                    st.warning("No tables were found in the selected pages.")
    
    except Exception as e:
        st.error("An issue occurred while processing the PDF. Please try again or try with a different PDF file.")
        logging.error(f"PDF processing error details: {str(e)}")
    
    finally:
        # Close the document before attempting to delete the file
        if 'doc' in locals() and doc:
            doc.close()
            
        # Clean up the temporary file with better error handling
        if os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except PermissionError:
                logging.warning(f"Could not delete temporary file {pdf_path} - it may still be in use")
            except Exception as e:
                logging.warning(f"Error deleting temporary file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='font-size: 12px;'>AIs can make mistakes, please review the output before using it for any purpose.</p>", unsafe_allow_html=True)
