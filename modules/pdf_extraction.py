import pymupdf  
import tkinter as tk
from tkinter import filedialog
import logging
import pymupdf 
import base64
import pandas as pd
import re
import ast
import asyncio
from modules.llm import table_identification_llm, vision_column_llm_parser, vision_llm_parser
import aiohttp


# Configure logging: You can adjust the level to DEBUG for more detailed output.
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more granular messages.
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def extract_text_from_pages(pdf_input, pages=None):
    """
    Extracts text from specified pages in a PDF file using PyMuPDF.
    
    Parameters:
        pdf_input (str or file-like object): The path to the PDF file or a file-like object.
        pages (int, list, tuple, or None): 
            - If an integer, extracts text from that specific page (0-indexed).
            - If a list of integers, extracts text from the specified pages.
            - If a tuple of two integers, treats it as a range (start, end) and extracts from start (inclusive)
              to end (exclusive).
            - If None, extracts text from all pages.
    
    Returns:
        str: The concatenated text extracted from the specified pages.
    """
    text = ""
    
    ## pages = pages-1 # adding this in as python starts from 0 put people don't

    # Open the PDF file using PyMuPDF.  
    # If pdf_input is a string, assume it's a file path.
    if isinstance(pdf_input, str):
        doc = pymupdf.open(pdf_input)
    else:
        # Otherwise, assume it's a file-like object (e.g., from Streamlit uploader)
        # Make sure it's at the beginning (reset pointer) and get its bytes.
        pdf_input.seek(0)
        pdf_bytes = pdf_input.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    
    total_pages = doc.page_count  # total number of pages in the document
    
    # Determine which pages to extract.
    if pages is None:
        # Extract text from all pages.
        page_indices = range(total_pages)
    elif isinstance(pages, int):
        if pages < 0 or pages >= total_pages:
            raise ValueError(f"Page index {pages} is out of range. Total pages: {total_pages}")
        page_indices = [pages]
    elif isinstance(pages, (list, tuple)):
        if isinstance(pages, tuple) and len(pages) == 2:
            start, end = pages
            if not (isinstance(start, int) and isinstance(end, int)):
                raise ValueError("Start and end values must be integers.")
            if start < 0 or end > total_pages or start >= end:
                raise ValueError("Invalid page range specified.")
            page_indices = range(start, end)
        else:
            page_indices = []
            for p in pages:
                if not isinstance(p, int):
                    raise ValueError("Page indices must be integers.")
                if p < 0 or p >= total_pages:
                    raise ValueError(f"Page index {p} is out of range. Total pages: {total_pages}")
                page_indices.append(p)
    else:
        raise ValueError("Parameter 'pages' must be an int, list, tuple, or None.")
    
    # Extract text from the specified pages.
    for i in page_indices:
        page = doc.load_page(i)
        page_text = page.get_text()
        text += f"\n\n--- Page {i + 1} ---\n\n" + page_text + "\n|-|+++|-|\n" # "\n|-|+++|-|\n" will be a delimiter ill use to seperate the pages where needed
    
    doc.close()
    return text



def select_pdf_file():
    """
    Opens a file dialog for the user to select a PDF file.

    Returns:
        str: The path to the selected PDF file, or an empty string if no file was selected.
    """
    logging.info("Opening file selection dialog.")
    # Create the Tkinter root window.
    root = tk.Tk()
    # Hide the main window.
    root.withdraw()
    # Set the root window to be topmost so the dialog appears in front.
    root.attributes("-topmost", True)
    
    # Open the file dialog for PDF files.
    pdf_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )
    
    # Destroy the root window now that we're done with it.
    root.destroy()
    
    if pdf_path:
        logging.info("Selected PDF file: %s", pdf_path)
    else:
        logging.info("No PDF file was selected.")
    return pdf_path


def get_page_pixel_data(pdf_path, page_no, dpi = 300, image_type = 'png'):
    # Open the PDF file
    doc = pymupdf.open(pdf_path)

    # Select the first page (pages are 0-indexed)
    page = doc[page_no]

    # Calculate zoom factor for 300 DPI (since 300/72 ≈ 4.17)
    zoom_factor = dpi / 72

    matrix = pymupdf.Matrix(zoom_factor, zoom_factor)

    # Render the page to an image (pixmap) using the transformation matrix
    pix = page.get_pixmap(matrix=matrix)

    png_data = pix.tobytes(image_type)

    base64_image = base64.b64encode(png_data).decode('utf-8')

    return base64_image


def extract_table_info(text):
    """
    Extracts table information from a pattern description string.
    
    Parameters:
        pattern_desc_from_image (str): Text containing table pattern description
        
    Returns:
        tuple: Contains:
            - num_tables (int): Number of tables found, or None if not found
            - table_headers (list): List of table header strings
            - parsing_rules (str): Rules for parsing tables and headers
    """
    
    # 1) Extract the Number of Tables
    match_num_tables = re.search(r'Number of Tables on the Page:\s*(\d+)', text)
    if match_num_tables:
        num_tables = int(match_num_tables.group(1))
    else:
        num_tables = None  # or handle error

    # 2) Extract the Table Headers
    #    We'll grab everything after "Table Headers:" until the line "3. Rules" begins
    match_headers = re.search(r'Table Headers:\s*(.*?)\s*\n\s*3\.', text, re.DOTALL)
    if match_headers:
        headers_text = match_headers.group(1)
        # Split on '||' to get individual headers
        table_headers = [h.strip() for h in headers_text.split('||')]
    else:
        table_headers = []


    # 3) Extract the Table Location
    match_location = re.search(r'Table Location for each table:\s*(.*)', text, re.DOTALL)
    if match_location:
        table_location = match_location.group(1).strip()
    else:
        table_location = ""

    table_location = table_location.split(" || ")

    return num_tables, table_headers, table_location


def compare_table_headers(headers1, headers2):
    """
    Compare two lists of table headers and return True if they are the same.
    """
    if len(headers1) != len(headers2):
        return False
    
    return all(h1.strip() == h2.strip() for h1, h2 in zip(headers1, headers2))



async def get_validated_table_info(text_input, open_api_key, base64_image, model='gpt-4o'):

    async def asycn_pattern_desc():

        pattern_desc = await table_identification_llm(
            text_input=text_input,
            base64_image=base64_image,
            open_api_key=open_api_key,
            model=model
        )
        return pattern_desc


    tasks = []
    # Create all tasks first
    async with asyncio.TaskGroup() as tg:
        tasks.append(tg.create_task(asycn_pattern_desc()))
        tasks.append(tg.create_task(asycn_pattern_desc()))
    
    # Get results after tasks complete
    output1 = await tasks[0]
    output2 = await tasks[1]
    
    num_tables1, headers1, table_location = extract_table_info(output1)
    num_tables2, headers2, table_location = extract_table_info(output2)
    print(num_tables1, num_tables2)
    if compare_table_headers(headers1, headers2) or (num_tables1 == num_tables2 and num_tables1 is not None):
        print('Initial headers match or same number of tables')
        return num_tables1, headers1, table_location, 0

    # Create third task if needed
    async with asyncio.TaskGroup() as tg:
        task3 = tg.create_task(asycn_pattern_desc())
    
    output3 = await task3
    num_tables3, headers3, table_location = extract_table_info(output3)
    
    # If headers match exactly or number of tables is the same, use first run results
    if compare_table_headers(headers3, headers1) or (num_tables3 == num_tables1 and num_tables3 is not None):
        print('Majority match found with first and third results')
        return num_tables1, headers1, table_location, 1
    
    # If headers match exactly or number of tables is the same, use first run results
    if compare_table_headers(headers3, headers2) or (num_tables3 == num_tables2 and num_tables3 is not None):
        print('Majority match found with second and third results')
        return num_tables2, headers2, table_location, 1

    # If no matches found, return the third run results 
    print('No matches found. Returning third run results for table_headers')
    return num_tables3, headers3, table_location, 2



def compare_column_data(data1, data2):
    issue_table_headers = []
    """
    Compare two sets of column data results and return True if they match.
    Compares column names for each index to ensure consistency.
    """
    if len(data1) != len(data2):
        logging.warning("Column data length mismatch")
        return False
    
    # Sort both lists by index to ensure we compare corresponding entries
    data1_sorted = sorted(data1, key=lambda x: x["index"])
    data2_sorted = sorted(data2, key=lambda x: x["index"])
    
    for item1, item2 in zip(data1_sorted, data2_sorted):
        # Compare column names for the same index (order doesn't matter)
        if set(item1["column_names"]) != set(item2["column_names"]):
            logging.warning(f"Column names mismatch for index {item1['index']}")
            logging.warning(f"Set 1: {item1['column_names']}")
            logging.warning(f"Set 2: {item2['column_names']}")
            issue_table_headers.append(item1["table_header"])
            return False, issue_table_headers
            
    return True, issue_table_headers


def extract_columns(response_text, tables_to_target):
    pattern = r'index:\s*\[(\d+)\].*?column_names:\s*\[(.*?)\].*?example_value_per_column:\s*\[(.*?)\].*?table_location:\s*\[(.*?)\]'
    matches = re.findall(pattern, response_text)
    
    results = []
    for index_str, columns_str, example_values_str, location_str in matches:
        index_value = int(index_str)
        columns_list = [col.strip().strip('"\'') for col in columns_str.split(',')]
        
        # Parse example values into a dictionary
        example_values = {}
        for pair in example_values_str.split(','):
            if ':' in pair:
                key, value = pair.split(':')
                key = key.strip().strip('"\'')
                value = value.strip().strip('()').strip('"\'')
                example_values[key] = value
        
        header = tables_to_target[index_value]
        
        results.append({
            "index": index_value,
            "table_header": header,
            "column_names": columns_list,
            "example_values_per_column": example_values,
            "table_location": location_str.strip()
        })
    return results


async def parse_column_data(user_text, text_input, tables_to_target, base64_image, table_list, open_api_key, model='gpt-4o'):
    async def async_pattern_desc():
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                llm_parser_desc = await vision_column_llm_parser(
                    user_text=user_text,
                    text_input=text_input,
                    table_to_target=tables_to_target,
                    base64_image=base64_image,
                    open_api_key=open_api_key,
                    model=model
                )
                return llm_parser_desc
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise  # Re-raise the last exception
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff

    tasks = []
    # Create all tasks first
    try:
        async with asyncio.TaskGroup() as tg:
            tasks.append(tg.create_task(async_pattern_desc()))
            tasks.append(tg.create_task(async_pattern_desc()))

        # Get results after tasks complete
        output1 = await tasks[0]
        output2 = await tasks[1]

        results1 = extract_columns(output1, table_list)
        results2 = extract_columns(output2, table_list)

        bool_decision, issue_table_headers_0 = compare_column_data(results1, results2)
        
        if bool_decision:
            logging.info("Initial phase column parsing match")
            return results1, issue_table_headers_0, 0  # Return score 0 for perfect match
        
        # Create third task if needed
        async with asyncio.TaskGroup() as tg:
            task3 = tg.create_task(async_pattern_desc())
        
        output3 = await task3
        results3 = extract_columns(output3, table_list)

        logging.warning(f"\n comparing results 1 and 3")
        bool_decision, issue_table_headers = compare_column_data(results1, results3)
        if bool_decision:
            logging.warning("Majority match found with first and third results")
            return results1, issue_table_headers_0, 1 # If the the majority matches if found then there will not be an issue table headers thus return the initial one that caused the issue in the first place. 
        
        logging.warning(f"\n comparing results 2 and 3")
        bool_decision, issue_table_headers = compare_column_data(results2, results3)
        if bool_decision:
            logging.warning("Majority match found with second and third results")
            return results2, issue_table_headers_0, 1
        
        # If no majority found, return third result with low confidence score
        logging.warning("No matching results found, returning third result with low confidence")
        return results3, issue_table_headers, 2

    except aiohttp.ClientError as e:
        logging.error(f"Network error occurred: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise



def parse_variable_data_to_df(text):
    """
    Parse variable data text into a pandas DataFrame.
    
    Args:
        text (str): Text containing variable data in format [column:val1 |-| val2 |-| ...]
        
    Returns:
        pd.DataFrame: DataFrame containing the parsed data with columns padded to equal length
    """
    pattern = r"\[([^\]:]+):([^]]+)\]"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    data = {}
    max_len = 0

    # Split each matched value on "|-|"
    for key, val in matches:
        items = [item.replace("***", "").strip()  # optionally remove "***"
                 for item in val.split("|-|")]
        data[key.strip()] = items
        max_len = max(max_len, len(items))

    # Build a DataFrame, ensuring all columns have the same length
    df = pd.DataFrame({
        col: values + [None]*(max_len - len(values))  # pad with None if needed
        for col, values in data.items()
    })

    return df


def extract_df_from_string(text):
    # Find the content between the first [ and last ], including the brackets
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        # Convert the matched string to a Python object using ast.literal_eval
        data = ast.literal_eval(match.group(1))
        # Create DataFrame from the data
        df = pd.DataFrame(data)
        return df
    raise ValueError("No tables extracted from the page")



async def process_tables_to_df(table_headers, table_location, user_text, extracted_text, base64_image, open_api_key, page_number):
    try:
        tasks = []
        # Create all tasks first 
        async with asyncio.TaskGroup() as tg:
            for table in table_headers:
                tasks.append(tg.create_task(vision_llm_parser(
                    user_text=user_text,
                    text_input=extracted_text,
                    table_to_target=table,
                    base64_image=base64_image,
                    open_api_key=open_api_key,
                    model='gpt-4o'
                )))

        # Await all tasks to complete
        results_output = []
        for task in tasks:
            results_output.append(await task)
            
    except Exception as e:
                
        # If first attempt fails, try again with model o1
        logging.warning(f"Initial table extraction failed: {str(e)}. Retrying with o1 model")
        try:
            tasks = []
            # Create all tasks first 
            async with asyncio.TaskGroup() as tg:
                for table in table_headers:
                    tasks.append(tg.create_task(vision_llm_parser(
                        user_text=user_text,
                        text_input=extracted_text,
                        table_to_target=table,
                        base64_image=base64_image,
                        open_api_key=open_api_key,
                        model='o1'
                    )))

            # Await all tasks to complete
            results_output = []
            for task in tasks:
                results_output.append(await task)
        except Exception as e:
            raise ValueError(f"Processing failed - table could not be extracted: {str(e)}")



    df_list = []
    for i, out in enumerate(results_output):

        df = extract_df_from_string(out)

        # Convert column names to strings first, then apply string operations
        df.columns = df.columns.astype(str).str.strip().str.strip('"\'').str.title()

        if table_location[i] == 'Table is present in both the image and the text document':
            # Apply mapping to all columns except table_header_descriptor
            df[df.columns] = df[df.columns].map(lambda val: val if str(val) in extracted_text else "N/A")
            df['table_header_descriptor'] = table_headers[i]
            df_list.append(df)
        else:
            df['table_header_descriptor'] = table_headers[i]
            df_list.append(df)


    df_out = pd.concat(df_list, ignore_index=True)

    unique_headers = df_out['table_header_descriptor'].unique()
    header_to_num = {header: f"Page {page_number + 1} Table {i+1}" for i, header in enumerate(unique_headers)}

    # Create new table column based on the mapping
    df_out['page_table'] = df_out['table_header_descriptor'].map(header_to_num)

    return df_out




# async def process_pdf_to_df(user_text, text_input, base64_image, open_api_key):
#     # Add await here
    
    # # Add await here since process_tables_to_df is async
    # df_final = await process_tables_to_df(
    #     table_headers, 
    #     table_location,
    #     user_text, 
    #     text_input, 
    #     base64_image, 
    #     open_api_key
    # )
    
    # return df_final


