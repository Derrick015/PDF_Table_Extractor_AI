import logging
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import asyncio
import os
import pymupdf
import time

from modules.pdf_extraction import (
    extract_text_from_pages,
    select_pdf_file,
    write_output_final,
    parse_column_data,
    get_validated_table_info,
    get_page_pixel_data,
    process_tables_to_df
)

# Create files directory if it doesn't exist
if not os.path.exists("files"):
    os.makedirs("files")

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure Logging
# Debug
# Info
# Warning
# Error
# Critical

log_file = os.path.join("logs", "main.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will continue to show logs in console
    ]
)


# Log the start of the script
logging.info("Starting PDF Table Extractor main script")


file_name = 'split_test_1'    
user_text = 'Extract all data from the table(s)'
table_in_image = True # may require more review
add_in_table_and_page_information = True


# 1. Load Credentials
logging.info("Loading environment variables from .env file.")
load_dotenv()

# Get the API key from the environment variable
open_api_key = os.getenv('OPENAI_API_KEY')
if not open_api_key:
    logging.warning("OPENAI_API_KEY is not set. Please check your .env file.")

logging.info("Initializing OpenAI client.")
openai_client = OpenAI(api_key=open_api_key)

# 2. Select PDF file and extract text
logging.info("Prompting user to select a PDF file.")
pdf_path = select_pdf_file()

if not pdf_path:
    logging.error("No PDF file was selected. Exiting.")
    raise SystemExit("No PDF file selected.")

logging.info(f"Opening PDF file: {pdf_path}")
doc = pymupdf.open(pdf_path)
total_pages = doc.page_count
page_indices = range(total_pages)
logging.info(f"Total pages in the document: {total_pages}")



# Start timing
start_time = time.time()

async def process_page():
    logging.info("Starting asynchronous page processing.")
    tasks = []
    results_output = []
    try:
        async with asyncio.TaskGroup() as tg:
            for page_no in page_indices:
                logging.debug(f"Loading page {page_no + 1}.")
                page = doc.load_page(page_no)
                extracted_text = page.get_text()
                

                if not table_in_image: 
                    # Check for the presence of tables with pymupdf. Only works for where there are not tables in the image. 
                    tabs = page.find_tables()
                    num_tables_0 = len(tabs.tables)
                    if num_tables_0 == 0:
                        print(f"No tables found on page from pymupdf {page_no + 1}, skipping...")
                        continue

                logging.debug(f"Converting page {page_no + 1} to base64 image.")
                base64_image = get_page_pixel_data(
                    pdf_path=pdf_path,
                    page_no=page_no,
                    dpi=500,
                    image_type='png'
                )
            
                logging.debug("Validating table information via LLM.")
                num_tables, table_headers, confidence_score_0 = await get_validated_table_info(
                    text_input=extracted_text,
                    user_text=user_text,
                    open_api_key=open_api_key,
                    base64_image=base64_image
                )

                if num_tables == 0:
                    logging.info(f"No tables found on page by LLM {page_no + 1}, skipping...")
                    continue

                logging.info(f"Found {num_tables} table(s) on page {page_no + 1}. Headers: {table_headers}")
        
                tasks.append(tg.create_task(process_tables_to_df(
                    table_headers,
                    user_text,
                    extracted_text,
                    base64_image,
                    open_api_key,
                    page_no,
                    table_in_image,
                    add_in_table_and_page_information
                )))
            
            # Await all tasks to complete
            logging.debug("Awaiting all table-processing tasks to finish.")
            for task in tasks:
                results_output.append(await task)
        
        if not results_output:
            logging.error("No tables found on any of the processed pages.")
            raise ValueError("No tables found on any of the processed pages")
            
        return results_output

    except Exception as e:
        logging.error("An issue occured. Kindly try again.")
        print(f"Error occurred during API call: {e}")

logging.info("Running the asynchronous process to parse PDF pages.")
output_final = asyncio.run(process_page())

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
logging.info(f"Processing took {elapsed_time:.2f} seconds")

# Save to Excel - combined tables (option=1)
output_path_combined = f'files/{file_name}_page_combined.xlsx'
logging.info(f"Writing combined table results to '{output_path_combined}'.")
write_output_final(output_final, excel_path=output_path_combined, option=1)
logging.info(f"Results saved to {output_path_combined}")

# Save to Excel - split tables (option=2)
output_path_split = f'files/{file_name}_page_split.xlsx'
logging.info(f"Writing split tables to '{output_path_split}'.")
write_output_final(output_final, excel_path=output_path_split, option=2)
logging.info(f"Results saved to {output_path_split}")

# Save to Excel - one sheet with gaps (option=3)
output_path_one_sheet = f'files/{file_name}_one_sheet_split.xlsx'
logging.info(f"Writing all tables on one sheet to '{output_path_one_sheet}'.")
write_output_final(output_final, excel_path=output_path_one_sheet, option=3)
logging.info(f"Results saved to {output_path_one_sheet}")

logging.info("All tasks completed successfully. Exiting main.")
