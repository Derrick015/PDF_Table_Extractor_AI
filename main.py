from modules.pdf_extraction import extract_text_from_pages, select_pdf_file
from modules.pdf_extraction import parse_column_data
from modules.pdf_extraction import get_validated_table_info
from modules.pdf_extraction import get_page_pixel_data
from modules.pdf_extraction import process_tables_to_df
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import asyncio
import os
import pymupdf
import time

# to do.. table names headers can be dupilcated may be helpful to use say table 1,2, etc to deitigusih tem. 
# use pdf plumber  page.find_tables() and gpt table count for confidence calculation. 
# the same table can be extraected twice or more if the header is not clear
# extract header witout the text 

file_name = 'test_8'    
user_text='Extract all data from the table(s) the header'

# 1. Load Credientials

# Load environment variables from the .env file
load_dotenv()
# Get the API key from the environment variable
open_api_key = os.getenv('OPENAI_API_KEY')
# Initialize OpenAI client
openai_client = OpenAI(api_key = open_api_key)





# 2. Select PDF file and extract text
pdf_path = select_pdf_file()
doc = pymupdf.open(pdf_path)
total_pages = doc.page_count  # total number of pages in the document
page_indices = range(total_pages)

# page_indices can be a list of page numbers to process


# Start timing
start_time = time.time()

async def process_page():
    tasks = []
    results_output = []
    # Create all tasks first 
    async with asyncio.TaskGroup() as tg:
        for page_no in page_indices:
            page = doc.load_page(page_no)
            extracted_text = page.get_text()
            
            # extracted_text = extract_text_from_pages(pdf_path, pages=page_no)
            base64_image = get_page_pixel_data(pdf_path=pdf_path, page_no=page_no, 
                                dpi = 500, image_type = 'png')
        
            num_tables, table_headers, table_location, confidence_score_0 = await get_validated_table_info(
                text_input=extracted_text, 
                open_api_key=open_api_key, 
                base64_image=base64_image
            )

            if num_tables == 0:
                print(f"No tables found on page {page_no + 1}, skipping...")
                continue
    
            tasks.append(tg.create_task(process_tables_to_df(
                table_headers, 
                table_location,
                user_text, 
                extracted_text, 
                base64_image, 
                open_api_key,
                page_no)))
            
        # Await all tasks to complete
        for task in tasks:
            results_output.append(await task)
    
    if not results_output:
        raise ValueError("No tables found on any of the processed pages")
            
    df_out_1 = pd.concat(results_output, ignore_index=True)
    
    return df_out_1


output_final = asyncio.run(process_page())

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing took {elapsed_time:.2f} seconds")

# Save to Excel
output_final.to_excel(f'files/{file_name}.xlsx', index=False)
print(f"Results saved to files/{file_name}.xlsx")


# num_tables, table_headers, confidence_score_0 = asyncio.run(get_validated_table_info(text_input=extracted_text, open_api_key=open_api_key, base64_image= base64_image))




# results, issue_table_headers_parse_column_data, confidence_score_1 =  asyncio.run(parse_column_data(user_text=user_text,
#                                         text_input= extracted_text,
#                                         tables_to_target= sorted(table_headers),
#                                         base64_image=base64_image,
#                                         open_api_key=open_api_key))





# df_final = process_tables_to_df(results, user_text, extracted_text, base64_image, open_api_key)
