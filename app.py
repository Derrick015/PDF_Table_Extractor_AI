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
import pandas as pd
import itertools
import re

from modules.pdf_extraction import (
    get_page_pixel_data,
    get_validated_table_info,
    process_tables_to_df,
    write_output_final,
    write_output_to_csv
)

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")


# Configure Logging
# Debug
# Info
# Warning
# Error
# Critical

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
# st.markdown("Upload a PDF file to extract tables.")

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
    
    st.markdown("---")  # Add some space with a horizontal line
    
    # Add checkbox for table in image detection
    table_in_image = st.checkbox("Image & Inference Mode", value=True, 
                                help="Enable this mode for: (1) Extracting tables from images within PDFs, (2) Adding creative interpretations like additional columns or values based on user instructions. Note: This mode bypasses text validation for more flexible results.")
    
    # Add checkbox to include table and page information in output
    add_in_table_and_page_information = st.checkbox("Add table and page information", value=False, 
                                 help="Enable this if you want to add table name, position and page number to the table")

    st.markdown("---")  # Add some space with a horizontal line
    
    # Model selection dropdown for AI processing
    model = st.selectbox(
        "Select AI model",
        options=["o1", "gpt-4o", "gpt-4o-mini",],
        index=1  # Default to gpt-4o as recommended option
    )
    
    # Display information about available models to help users make appropriate selection
    st.markdown("""
        <div style="font-size:0.8em; color:gray;">
        <strong>Model information:</strong><br>
        • <strong>o1</strong>: Advanced but expensive model, may handle complex layouts better<br>
        • <strong>gpt-4o</strong>: Balanced performance, recommended for most tables<br>
        • <strong>gpt-4o-mini</strong>: Faster, lower cost, but may be less accurate for complex tables<br>

        </div>
    """, unsafe_allow_html=True)

    # Add horizontal line for visual separation of sections
    st.markdown("---")
    
    # Output format selection section
    st.subheader("Output Format")
    file_format = st.selectbox(
        "Select file format:",
        options=["Excel (.xlsx)", "CSV (.csv)"],
        index=0  # Default to Excel format
    )
    
    # Input validation logic - ensure sensible defaults if user inputs are empty
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
    
    # Load PDF document and validate it can be opened
    try:
        doc = pymupdf.open(pdf_path)
        total_pages = doc.page_count
        
        st.success(f"Successfully loaded PDF with {total_pages} pages.")
        
        # Page range selection section - allows users to choose which pages to process
        st.subheader("Page Range Selection")
        range_option = st.radio("Select pages to process:", 
                               ["All pages", "Specific range", "Custom pages"])
        
        if range_option == "All pages":
            # Process the entire document
            page_indices = list(range(total_pages))
            st.info(f"Processing all {total_pages} pages")
            
        elif range_option == "Specific range":
            # Allow selection of a continuous range of pages
            col1, col2 = st.columns(2)
            
            # Initialize end_page in session state if it doesn't exist
            # This preserves the value between reruns of the Streamlit app
            if 'end_page' not in st.session_state:
                st.session_state.end_page = min(5, total_pages)  # Default to page 5 or max
                
            with col1:
                # Start page selection with input validation
                start_page = st.number_input("Start page", min_value=1, max_value=total_pages, value=1, key="start_page")
            
            with col2:
                # End page selection with dynamic minimum value based on start page
                # Ensures end page is always >= start page
                if 'end_page' not in st.session_state:
                    st.session_state.end_page = min(5, total_pages)  # Default to page 5 or max
                elif st.session_state.end_page < start_page:
                    st.session_state.end_page = start_page
                
                end_page = st.number_input(
                    "End page", 
                    min_value=start_page, 
                    max_value=total_pages, 
                    key="end_page"
                )
            
            # Convert 1-indexed user input to 0-indexed page indices for processing
            page_indices = list(range(start_page - 1, end_page))
            st.info(f"Processing pages {start_page} to {end_page} (total: {len(page_indices)} pages)")
            
            # Display preview of start and end pages to help users verify selection
            st.subheader("Range Preview")
            preview_col1, preview_col2 = st.columns(2)
            
            with preview_col1:
                st.markdown(f"**Start Page ({start_page})**")
                start_page_index = start_page - 1
                # Render the start page preview image
                page = doc.load_page(start_page_index)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # Scale up for better visibility
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption=f"Page {start_page}", use_container_width=True)
            
            with preview_col2:
                st.markdown(f"**End Page ({end_page})**")
                end_page_index = end_page - 1
                # Render the end page preview image
                page = doc.load_page(end_page_index)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # Scale up for better visibility
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption=f"Page {end_page}", use_container_width=True)
            
        else:  # Custom pages option
            # Allow selection of non-consecutive pages using comma-separated list
            custom_pages = st.text_input("Enter page numbers separated by commas (e.g., 1,3,5,8)")
            preview_button = st.button("Preview Pages")
            
            # Process and preview custom pages when requested or when using previously entered values
            if custom_pages and (preview_button or 'custom_pages_last' in st.session_state and st.session_state.custom_pages_last == custom_pages):
                try:
                    # Store current custom pages value in session state to maintain preview between interactions
                    st.session_state.custom_pages_last = custom_pages
                    
                    # Parse and validate the page numbers entered by the user
                    page_nums = [int(p.strip()) for p in custom_pages.split(",")]
                    # Filter out invalid page numbers
                    valid_pages = [p for p in page_nums if 1 <= p <= total_pages]
                    page_indices = [p - 1 for p in valid_pages]  # Convert to 0-based indices for internal use
                    
                    # Warn if some entered page numbers were invalid
                    if len(valid_pages) != len(page_nums):
                        st.warning(f"Some page numbers were out of range and will be ignored. Valid range: 1-{total_pages}")
                    
                    st.info(f"Processing {len(page_indices)} pages: {', '.join(map(str, valid_pages))}")
                    
                    # Display previews of custom pages (up to 4 to avoid overcrowding the UI)
                    if valid_pages:
                        st.subheader("Page Previews")
                        preview_pages = valid_pages[:4]  # Show max 4 previews
                        
                        # Create a dynamic number of columns based on the preview pages
                        columns = st.columns(min(len(preview_pages), 4))
                        for i, page_num in enumerate(preview_pages):
                            with columns[i]:
                                st.markdown(f"**Page {page_num}**")
                                # Render the preview for each selected page
                                page = doc.load_page(page_num - 1)
                                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                                img_bytes = pix.tobytes("png")
                                st.image(img_bytes, caption=f"Page {page_num}", use_container_width=True)
                        
                        # Indicate if not all selected pages are shown in the preview
                        if len(valid_pages) > 4:
                            st.info(f"Showing first 4 of {len(valid_pages)} selected pages")
                except ValueError:
                    # Handle invalid input (non-numeric values)
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
                
                # Show progress indicators for the processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                async def process_pages():
                    """
                    Asynchronously processes PDF pages to extract tables.
                    
                    This function:
                    1. Creates tasks for each selected page
                    2. Extracts both text and image data from each page
                    3. Validates and identifies tables using AI
                    4. Processes identified tables into DataFrame objects
                    5. Updates progress indicators throughout the process
                    6. Handles errors and retries when necessary
                    
                    Returns a list of processed table data
                    """
                    tasks = []
                    results_output = []
                    
                    try:
                        status_text.text("Initializing page processing...")
                        
                        # Process each page concurrently using async TaskGroup
                        async with asyncio.TaskGroup() as tg:
                            for i, page_no in enumerate(page_indices):
                                status_text.text(f"Processing page {page_no + 1}...")
                                progress_bar.progress((i / len(page_indices)) * 0.5)  # Update to 50% through the process
                                
                                # Load the current page from the document
                                page = doc.load_page(page_no)
                                
                                if not table_in_image:
                                # Check for tables using PyMuPDF's table detection
                                # If table_in_image is False, skip pages with no tables detected by PyMuPDF
                                    tabs = page.find_tables()
                                    num_tables_0 = len(tabs.tables)
                                    
                                    if num_tables_0 == 0:
                                        st.info(f"No tables found on page {page_no + 1}")
                                        continue
                                
                                # Extract text content from the page
                                extracted_text = page.get_text()
                                
                                # Convert the page to an image for AI vision processing
                                base64_image = get_page_pixel_data(
                                    pdf_path=pdf_path,
                                    page_no=page_no,
                                    dpi=500,
                                    image_type='png'
                                )
                                
                                # Use AI to identify and validate tables on the page
                                num_tables, table_headers, _ = await get_validated_table_info(
                                    text_input=extracted_text,
                                    user_text=user_text,
                                    open_api_key=open_api_key,
                                    base64_image=base64_image,
                                    model='gpt-4o' # for table header detection stick with gpt-4o
                                )
                                
                                logging.debug(f"num_tables: {num_tables}")
                                logging.debug(f"table_headers: {table_headers}")


                                # Check for the presence of tables with LLM. 
                                if num_tables == 0:
                                    st.info(f"No tables found on page {page_no + 1}")
                                    continue
                                
                                
                                # Create an asynchronous task for processing each table
                                # This allows concurrent processing of tables across multiple pages
                                tasks.append(tg.create_task(process_tables_to_df(
                                    table_headers,
                                    user_text,
                                    extracted_text,
                                    base64_image,
                                    open_api_key,
                                    page_no,
                                    table_in_image,
                                    add_in_table_and_page_information,
                                    model
                                )))
                            
                            # Await all tasks to complete and collect results
                            for j, task in enumerate(tasks):
                                results_output.append(await task)
                                progress_bar.progress(0.5 + ((j + 1) / len(tasks)) * 0.5)  # Update from 50% to 100%
                        
                        status_text.text("Processing complete!")
                        return results_output
                        
                    except Exception as e:
                        # Handle any errors during processing
                        st.error("An issue occurred during processing. Please try again. If the issue persists, try with a different page range or check your PDF file.")
                        logging.error(f"Processing error details: {str(e)}")
                        return []
                
                # Start the asynchronous processing workflow
                start_time = time.time()
                with st.spinner("Processing PDF tables..."):
                    # Run the async function in the main thread
                    output_final = asyncio.run(process_pages())
                    # Store the output in session state for persistence between Streamlit reruns
                    st.session_state.output_final = output_final
                    st.session_state.processing_complete = True
                
                # Calculate and display processing time for performance feedback
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.success(f"Processing completed in {elapsed_time:.2f} seconds")
            
            # Results display section - shows after processing is complete
            if st.session_state.processing_complete:
                output_final = st.session_state.output_final
                
                # Only show preview and download options if we have results
                if output_final and len(output_final) > 0:
                    # Add a toggle to control preview visibility
                    # This helps manage UI complexity for large results
                    show_preview = st.checkbox("Show data preview", value=True)
                    
                    if show_preview:
                        # Let user select how to format the preview
                        # Different formats are useful for different use cases
                        preview_format = st.selectbox(
                            "Select preview format:",
                            options=["Format 1: All tables concatenated", 
                                    "Format 2: Tables by page", 
                                    "Format 3: All tables on one sheet"],
                            index=2  # Default to Format 3
                        )
                        
                        # Add download button for the currently selected format
                        # Make sure the files directory exists
                        if not os.path.exists("files"):
                            os.makedirs("files")
                            
                        # Create the download button based on file format
                        if file_format == "Excel (.xlsx)":
                            # Get the format option from the radio button selection
                            format_option = int(preview_format.split(":")[0].split(" ")[1])
                            excel_file = f'files/{file_name}_format_{format_option}.xlsx'
                            write_output_final(output_final, excel_path=excel_file, option=format_option)
                            
                            with open(excel_file, "rb") as file:
                                st.download_button(
                                    label="Download",
                                    data=file,
                                    file_name=f"{file_name}_{preview_format.split(':')[0].strip().replace(' ', '_').lower()}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        else:  # CSV format
                            csv_base_path = f'files/{file_name}'
                            format_option = int(preview_format.split(":")[0].split(" ")[1])
                            
                            if format_option == 1:  # Format 1: All tables concatenated
                                csv_file = f'{csv_base_path}_concatenated.csv'
                                write_output_to_csv(output_final, csv_base_path=csv_base_path, option=1)
                                
                                with open(csv_file, "rb") as file:
                                    st.download_button(
                                        label="Download",
                                        data=file,
                                        file_name=f"{file_name}_concatenated.csv",
                                        mime="text/csv"
                                    )
                            elif format_option == 2:  # Format 2: Tables by page
                                # For CSV format 2, we create a zip with multiple files
                                csv_files = write_output_to_csv(output_final, csv_base_path=csv_base_path, option=2)
                                
                                if csv_files:
                                    # Create a zip file for multiple CSV files
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                        for csv_path in csv_files:
                                            filename = os.path.basename(csv_path)
                                            with open(csv_path, "rb") as f:
                                                zip_file.writestr(filename, f.read())
                                    
                                    # Set buffer position to start
                                    zip_buffer.seek(0)
                                    
                                    st.download_button(
                                        label="Download",
                                        data=zip_buffer,
                                        file_name=f"{file_name}_pages.zip",
                                        mime="application/zip"
                                    )
                            else:  # Format 3: All tables on one sheet
                                csv_file = f'{csv_base_path}_all_tables_with_gaps.csv'
                                write_output_to_csv(output_final, csv_base_path=csv_base_path, option=3)
                                
                                with open(csv_file, "rb") as file:
                                    st.download_button(
                                        label="Download",
                                        data=file,
                                        file_name=f"{file_name}_all_tables_with_gaps.csv",
                                        mime="text/csv"
                                    )
                        
                        # Get all DataFrames
                        all_dfs = list(itertools.chain.from_iterable(output_final))
                        
                        if preview_format == "Format 1: All tables concatenated":
                            # Format 1: All tables concatenated vertically
                            st.markdown("**Preview of 'All tables concatenated' format:**")
                            
                            if all_dfs:
                                # Limit to first 100 rows for preview
                                merged_df = pd.concat(all_dfs, axis=0)
                                preview_rows = min(100, len(merged_df))
                                st.dataframe(merged_df.head(preview_rows), use_container_width=True)
                                if len(merged_df) > preview_rows:
                                    st.info(f"Showing first {preview_rows} rows out of {len(merged_df)} total rows. Download the file to see all data.")
                            else:
                                st.info("No tables found to preview.")
                                
                        elif preview_format == "Format 2: Tables by page":
                            # Format 2: Tables by page
                            st.markdown("**Preview of 'Tables by page' format:**")
                            
                            # Create tabs for each page
                            if output_final:
                                # Limit to first 5 pages for preview
                                preview_pages = min(5, len(output_final))
                                tabs = st.tabs([f"Page {i+1}" for i in range(preview_pages)])
                                
                                for i in range(preview_pages):
                                    with tabs[i]:
                                        if output_final[i]:  # If page has tables
                                            for j, df in enumerate(output_final[i]):
                                                st.markdown(f"**Table {j+1}**")
                                                st.dataframe(df, use_container_width=True)
                                                if j < len(output_final[i]) - 1:
                                                    st.markdown("---")
                                        else:
                                            st.info("No tables found on this page.")
                                
                                if len(output_final) > 5:
                                    st.info(f"Showing first 5 pages out of {len(output_final)} total pages. Download the file to see all pages.")
                            else:
                                st.info("No pages with tables found to preview.")
                                
                        else:  # Format 3: All tables on one sheet
                            # Format 3: All tables on one sheet with gaps
                            st.markdown("**Preview of 'All tables on one sheet' format:**")
                            
                            # Limit to first 5 tables for preview
                            preview_dfs = all_dfs[:min(5, len(all_dfs))]
                            
                            # Display each table with a separator
                            for i, df in enumerate(preview_dfs):
                                st.markdown(f"**Table {i+1}**")
                                st.dataframe(df, use_container_width=True)
                                
                                # Add a separator between tables (except after the last one)
                                if i < len(preview_dfs) - 1:
                                    st.markdown("---")
                            
                            # Show a message if there are more tables
                            if len(all_dfs) > 5:
                                st.info("Download the file to see all tables.")
                    
                    # Add a horizontal line between preview and download sections
                    st.markdown("---")
                    

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
