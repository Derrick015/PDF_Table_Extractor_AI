import pymupdf
import tkinter as tk
from tkinter import filedialog
import logging
import pymupdf
import base64
import pandas as pd
import re
import ast
import itertools
import asyncio
from modules.llm import table_identification_llm, vision_column_llm_parser, vision_llm_parser
import aiohttp

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
    logging.info("Starting text extraction from PDF.")
    logging.debug(f"Received pdf_input={pdf_input}, pages={pages}")

    text = ""

    # Open the PDF file using PyMuPDF.  
    if isinstance(pdf_input, str):
        logging.debug(f"Opening PDF file from path: {pdf_input}")
        doc = pymupdf.open(pdf_input)
    else:
        logging.debug("Opening PDF file from file-like object.")
        pdf_input.seek(0)
        pdf_bytes = pdf_input.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    total_pages = doc.page_count
    logging.debug(f"PDF has {total_pages} pages.")

    # Determine which pages to extract.
    if pages is None:
        page_indices = range(total_pages)
    elif isinstance(pages, int):
        if pages < 0 or pages >= total_pages:
            logging.error(f"Page index {pages} is out of range. Total pages: {total_pages}")
            raise ValueError(f"Page index {pages} is out of range. Total pages: {total_pages}")
        page_indices = [pages]
    elif isinstance(pages, (list, tuple)):
        if isinstance(pages, tuple) and len(pages) == 2:
            start, end = pages
            if not (isinstance(start, int) and isinstance(end, int)):
                logging.error("Start and end values must be integers.")
                raise ValueError("Start and end values must be integers.")
            if start < 0 or end > total_pages or start >= end:
                logging.error("Invalid page range specified.")
                raise ValueError("Invalid page range specified.")
            page_indices = range(start, end)
        else:
            page_indices = []
            for p in pages:
                if not isinstance(p, int):
                    logging.error("Page indices must be integers.")
                    raise ValueError("Page indices must be integers.")
                if p < 0 or p >= total_pages:
                    logging.error(f"Page index {p} is out of range. Total pages: {total_pages}")
                    raise ValueError(f"Page index {p} is out of range. Total pages: {total_pages}")
                page_indices.append(p)
    else:
        logging.error("Parameter 'pages' must be an int, list, tuple, or None.")
        raise ValueError("Parameter 'pages' must be an int, list, tuple, or None.")

    # Extract text from the specified pages.
    for i in page_indices:
        logging.debug(f"Extracting text from page {i + 1}")
        page = doc.load_page(i)
        page_text = page.get_text()
        text += f"\n\n--- Page {i + 1} ---\n\n" + page_text + "\n|-|+++|-|\n"
    
    doc.close()
    logging.info("Completed text extraction.")
    return text

def select_pdf_file():
    """
    Opens a file dialog for the user to select a PDF file.

    Returns:
        str: The path to the selected PDF file, or an empty string if no file was selected.
    """
    logging.info("Opening file selection dialog.")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    pdf_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )
    
    root.destroy()
    
    if pdf_path:
        logging.info(f"Selected PDF file: {pdf_path}")
    else:
        logging.info("No PDF file was selected.")
    return pdf_path

def get_page_pixel_data(pdf_path, page_no, dpi=500, image_type='png'):
    """
    Converts a specified PDF page to a base64-encoded image.

    Parameters:
        pdf_path (str): The path to the PDF file
        page_no (int): Page number (0-indexed)
        dpi (int): Resolution in dots per inch
        image_type (str): Image file format ('png', 'jpeg', etc.)

    Returns:
        str: Base64-encoded image representation
    """
    logging.info(f"Converting PDF page {page_no + 1} to base64 image. DPI={dpi}, Format={image_type}")
    doc = pymupdf.open(pdf_path)
    page_count = doc.page_count
    if page_no >= page_count or page_no < 0:
        logging.error(f"Page number {page_no} out of range. Total pages: {page_count}")
        raise ValueError(f"Page number {page_no} out of range. Total pages: {page_count}")
    page = doc[page_no]

    zoom_factor = dpi / 72
    matrix = pymupdf.Matrix(zoom_factor, zoom_factor)

    pix = page.get_pixmap(matrix=matrix)
    png_data = pix.tobytes(image_type)
    base64_image = base64.b64encode(png_data).decode('utf-8')

    doc.close()
    logging.info("Finished converting page to base64.")
    return base64_image

def extract_table_info(text):
    """
    Extracts table information from a pattern description string.
    
    Parameters:
        text (str): Text containing table pattern description
        
    Returns:
        tuple: (num_tables, table_headers)
    """
    logging.debug("Extracting table info from text.")

    match_num_tables = re.search(r'Number of Tables on the Page:\s*(\d+)', text)
    if match_num_tables:
        num_tables = int(match_num_tables.group(1))
        logging.debug(f"Found number of tables: {num_tables}")
    else:
        num_tables = None
        logging.debug("No table count found in text.")

    # Modified regex to capture table headers without requiring "3." after
    match_headers = re.search(r'Table Headers:\s*(.*?)(?:\s*\n\s*3\.|$)', text, re.DOTALL)
    if match_headers:
        headers_text = match_headers.group(1).strip()
        # Remove any extra quotes and whitespace
        table_headers = [h.strip().strip('"') for h in headers_text.split('||')]
        logging.debug(f"Extracted table headers: {table_headers}")
    else:
        table_headers = []
        logging.debug("No table headers found.")

    return num_tables, table_headers

def compare_table_headers(headers1, headers2):
    """
    Compare two lists of table headers and return True if they are the same.
    """
    logging.debug(f"Comparing table headers:\n{headers1}\n{headers2}")
    if len(headers1) != len(headers2):
        logging.debug("Header length mismatch.")
        return False
    
    same = all(h1.strip() == h2.strip() for h1, h2 in zip(headers1, headers2))
    logging.debug(f"Headers are the same: {same}")
    return same

async def get_validated_table_info(text_input, user_text, open_api_key, base64_image, model='gpt-4o'):
    """
    Attempt to retrieve consistent table information by making multiple calls
    to the table identification LLM. If there's a majority match or exact match
    between attempts, return that; otherwise return the third attempt's output.
    """
    logging.info("Validating table information with multiple LLM calls.")

    async def asycn_pattern_desc():
        return await table_identification_llm(
            text_input=text_input,
            user_text=user_text,
            base64_image=base64_image,
            open_api_key=open_api_key,
            model=model
        )

    tasks = []
    # Create first two tasks
    async with asyncio.TaskGroup() as tg:
        tasks.append(tg.create_task(asycn_pattern_desc()))
        tasks.append(tg.create_task(asycn_pattern_desc()))
    
    # Wait for first two tasks
    output1 = await tasks[0]
    output2 = await tasks[1]
    logging.debug(f"LLM attempt 1 output:\n{output1}")
    logging.debug(f"LLM attempt 2 output:\n{output2}")

    num_tables1, headers1 = extract_table_info(output1)
    num_tables2, headers2 = extract_table_info(output2)

    # logging.debug(f"headers1: {headers1}")
    # logging.debug(f"headers2: {headers2}")
    # logging.debug(f"num_tables1: {num_tables1}")
    # logging.debug(f"num_tables2: {num_tables2}")

    if compare_table_headers(headers1, headers2) or (num_tables1 == num_tables2 and num_tables1 is not None):
        logging.info("Initial table info match or same table count. Returning first attempt's result.")
        return num_tables1, headers1, 0 # 0 indicates the highest confidence. The higher the number, the lower the confidence. 

    # Create third task if needed
    async with asyncio.TaskGroup() as tg:
        task3 = tg.create_task(asycn_pattern_desc())
    output3 = await task3
    logging.debug(f"LLM attempt 3 output:\n{output3}")

    num_tables3, headers3  = extract_table_info(output3)

    logging.debug(f"headers3: {headers3}")
    logging.debug(f"num_tables3: {num_tables3}")

    if compare_table_headers(headers3, headers1) or (num_tables3 == num_tables1 and num_tables3 is not None):
        logging.info("Majority match found with first and third results.")
        return num_tables1, headers1, 1
    
    if compare_table_headers(headers3, headers2) or (num_tables3 == num_tables2 and num_tables3 is not None):
        logging.info("Majority match found with second and third results.")
        return num_tables2, headers2, 1

    logging.warning("No matches found. Returning third run results for table_headers.")
    return num_tables3, headers3, 2

def compare_column_data(data1, data2):
    """
    Compare two sets of column data results and return (bool, issue_table_headers).
    If mismatch occurs, return which table headers encountered an issue.
    """
    logging.debug("Comparing column data for consistency.")
    issue_table_headers = []

    if len(data1) != len(data2):
        logging.warning("Column data length mismatch")
        return False, issue_table_headers
    
    data1_sorted = sorted(data1, key=lambda x: x["index"])
    data2_sorted = sorted(data2, key=lambda x: x["index"])
    
    for item1, item2 in zip(data1_sorted, data2_sorted):
        # Compare column names for the same index
        if set(item1["column_names"]) != set(item2["column_names"]):
            logging.warning(f"Column names mismatch for index {item1['index']}")
            logging.warning(f"Set 1: {item1['column_names']}")
            logging.warning(f"Set 2: {item2['column_names']}")
            issue_table_headers.append(item1["table_header"])
            return False, issue_table_headers
            
    logging.debug("Column data match found.")
    return True, issue_table_headers

def extract_columns(response_text, tables_to_target):
    """
    Extract column info from the LLM response text using regex. 
    """
    logging.debug("Extracting columns from LLM response.")
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
    logging.debug(f"Extracted columns result: {results}")
    return results

async def parse_column_data(user_text, text_input, tables_to_target, base64_image, table_list, open_api_key, model='gpt-4o'):
    """
    Asynchronously parse column data from the PDF using an LLM, making multiple attempts for reliability.
    """
    logging.info("Parsing column data with multiple attempts to ensure accuracy.")

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
                if attempt == max_retries - 1:
                    logging.error("All retries failed in async_pattern_desc. Raising exception.")
                    raise
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                await asyncio.sleep(retry_delay * (attempt + 1))

    tasks = []
    try:
        # First two attempts concurrently
        async with asyncio.TaskGroup() as tg:
            tasks.append(tg.create_task(async_pattern_desc()))
            tasks.append(tg.create_task(async_pattern_desc()))

        output1 = await tasks[0]
        output2 = await tasks[1]
        logging.debug(f"Output 1:\n{output1}")
        logging.debug(f"Output 2:\n{output2}")

        results1 = extract_columns(output1, table_list)
        results2 = extract_columns(output2, table_list)

        bool_decision, issue_table_headers_0 = compare_column_data(results1, results2)
        
        if bool_decision:
            logging.info("Initial phase column parsing match. Returning results1.")
            return results1, issue_table_headers_0, 0
        
        # Third attempt if needed
        async with asyncio.TaskGroup() as tg:
            task3 = tg.create_task(async_pattern_desc())
        output3 = await task3
        logging.debug(f"Output 3:\n{output3}")

        results3 = extract_columns(output3, table_list)

        logging.warning("Comparing results 1 and 3.")
        bool_decision, issue_table_headers = compare_column_data(results1, results3)
        if bool_decision:
            logging.warning("Majority match found with first and third results.")
            return results1, issue_table_headers_0, 1

        logging.warning("Comparing results 2 and 3.")
        bool_decision, issue_table_headers = compare_column_data(results2, results3)
        if bool_decision:
            logging.warning("Majority match found with second and third results.")
            return results2, issue_table_headers_0, 1

        logging.warning("No matching results found, returning third result with low confidence.")
        return results3, issue_table_headers, 2

    except aiohttp.ClientError as e:
        logging.error(f"Network error occurred: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in parse_column_data: {str(e)}")
        raise

def parse_variable_data_to_df(text):
    """
    Parse variable data text into a pandas DataFrame.
    """
    logging.info("Parsing variable data into DataFrame.")
    pattern = r"\[([^\]:]+):([^]]+)\]"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    data = {}
    max_len = 0

    for key, val in matches:
        items = [item.replace("***", "").strip() for item in val.split("|-|")]
        data[key.strip()] = items
        max_len = max(max_len, len(items))

    df = pd.DataFrame({
        col: values + [None]*(max_len - len(values))
        for col, values in data.items()
    })

    logging.debug(f"Variable data DataFrame shape: {df.shape}")
    return df

def extract_df_from_string(text):
    """
    Extracts a DataFrame from a string that contains a Python list/dict-like structure.
    """
    logging.debug("Extracting DataFrame from string representation.")
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        data = ast.literal_eval(match.group(1))
        df = pd.DataFrame(data)
        logging.debug(f"Extracted DataFrame shape: {df.shape}")
        return df
    logging.error("No tables extracted from the string.")
    raise ValueError("No tables extracted from the page")


async def process_tables_to_df(
    table_headers, 
    user_text, 
    extracted_text, 
    base64_image, 
    open_api_key, 
    page_number,
    table_in_image,
    add_in_table_and_page_information,  
    max_retries=3,
    initial_delay=1,
    backoff_factor=2,
    max_extract_retries_for_extraction_failures=2,
    model='gpt-4o'
):
    """
    Process tables by calling an LLM parser with exponential backoff.
    """
    logging.info(f"Processing tables to DataFrame for page {page_number + 1}")
    

    # 1) Try first model: 'gpt-4o'
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            logging.debug(f"[Model {model}] Attempt {attempt+1} of {max_retries}. Delay={delay}")
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for table in table_headers:
                    tasks.append(tg.create_task(
                        vision_llm_parser(
                            user_text=user_text,
                            text_input=extracted_text,
                            table_to_target=table,
                            base64_image=base64_image,
                            open_api_key=open_api_key,
                            model= model,
                            temperature=0.7
                        )
                    ))
            results_output = [task.result() for task in tasks]

            logging.info(f"Successfully retrieved data using model '{model}'.")
            break
        except Exception as e:
            logging.warning(
                f"[Model {model}] Attempt {attempt+1} of {max_retries} failed: {e}. "
                f"Retrying in {delay} second(s)..."
            )
            if attempt == max_retries - 1:
                logging.warning(f"Max retries with '{model}' exhausted.")
            else:
                await asyncio.sleep(delay)
                delay *= backoff_factor

    # 2) Process the results into DataFrames
    logging.debug(f"Comparing results ouput {len(results_output)} with the table headers {len(table_headers) } for page {page_number + 1}")
    # logging.debug(f"results_output: {results_output}")
    df_list = []
    for i, out in enumerate(results_output):
        extract_retry_count = 0

        max_extract_retries = max_extract_retries_for_extraction_failures  # Maximum number of retries for extraction failures
        
        while extract_retry_count <= max_extract_retries:
            try:
                # test to see if the LLM is returning the correct data. 

                df = extract_df_from_string(out)
                logging.debug(f"Parsed DataFrame for table index {i} with shape {df.shape}")

                # Normalize columns
                df.columns = df.columns.astype(str).str.strip().str.strip('"\'').str.title()
                
                if not table_in_image:
                    # Replace any values that are not in the extracted text with "N/A"  
                    df[df.columns] = df[df.columns].map(
                        lambda val: val if str(val) in extracted_text else "N/A"
                    )
                    
                if add_in_table_and_page_information:
                    df['table_header_position'] = table_headers[i]
                    df['page_number'] = page_number + 1

                df_list.append(df)
                break  # Successfully extracted, exit the retry loop
            
            except Exception as e:
                extract_retry_count += 1
                if extract_retry_count <= max_extract_retries:
                    logging.warning(f"Could not extract table with index {i} on page {page_number + 1}. Retry attempt {extract_retry_count}...")
                    # logging.error("Full error traceback:", exc_info=True)
                    # Regenerate the specific table result using vision_llm_parser. 
                    # Previous llm generation might not have been suitbale for the extraction function. 
                    try:
                        logging.info(f"Regenerating table data for index {i}, table '{table_headers[i]}'")
                        out = await vision_llm_parser(
                            user_text=user_text,
                            text_input=extracted_text,
                            table_to_target=table_headers[i],
                            base64_image=base64_image,
                            open_api_key=open_api_key,
                            model=model,
                            temperature=1
                        )
                        results_output[i] = out  # Update the results_output with the new result
                        logging.info(f"Regenerated table data for index {i}, with model 'o1', table '{table_headers[i]}, output was {out}")
                    except Exception as regen_error:
                        logging.error(f"Failed to regenerate table data: {str(regen_error)}")
                        # Continue to next retry or exit loop if max retries reached
                else:
                    logging.warning(f"Could not extract table with index {i} on page {page_number + 1} after {max_extract_retries} retries, skipping.")
                    break  # Exit the retry loop after max retries

    logging.info(f"Completed processing tables to DataFrame for page {page_number + 1}.")
    
    # Handle case where all tables failed extraction
    if not df_list:
        logging.error(f"No tables could be extracted from the results. - Page {page_number + 1}")
        df_list.append(pd.DataFrame()) # apend empty df

    return df_list

def sanitize_worksheet_name(name):
    """
    Sanitize Excel worksheet names by removing or replacing characters that are not allowed.
    
    Excel worksheet naming rules:
    - Can't exceed 31 characters
    - Can't contain: [ ] : * ? / \
    - Can't be 'History' as it's a reserved name
    
    Args:
        name (str): The original worksheet name
        
    Returns:
        str: Sanitized worksheet name safe for Excel
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[\[\]:*?/\\]', '_', str(name))
    
    # Truncate to 31 characters (Excel limit)
    if len(sanitized) > 31:
        sanitized = sanitized[:31]
        
    # Make sure it's not empty or 'History' (reserved name)
    if not sanitized or sanitized.lower() == 'history':
        sanitized = 'Sheet1'
        
    return sanitized

def write_output_final(output_final, excel_path, option=1, gap_rows=2):
    """
    Writes nested lists of DataFrames (`output_final`) to Excel in 3 different ways.

    :param output_final: A list of lists of DataFrames. 
    :param excel_path: Output Excel filename/path
    :param option: Choose 1 of 3 write modes:
                   1 = Horizontally merge (side-by-side) all DataFrames into one wide table (one sheet)
                   2 = Each top-level group on its own sheet, with `gap_rows` blank rows between sub-DataFrames
                   3 = Flatten all DataFrames onto one sheet vertically, with `gap_rows` blank rows between them
    :param gap_rows: How many blank rows to insert between tables (used in options 2 and 3).
    """
    logging.info(f"Writing output to Excel at '{excel_path}' with option={option}.")
    
    # Replace problematic characters in all dataframes before writing
    def sanitize_dataframe(df):
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Replace problematic characters in column names
        df_clean.columns = [re.sub(r'[\[\]:*?/\\]', '_', str(col)) for col in df_clean.columns]
        
        # Replace problematic characters in string data
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # Only process string columns
                # Replace problematic characters with underscores
                df_clean[col] = df_clean[col].astype(str).apply(
                    lambda x: re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', x) if pd.notna(x) else x
                )
        
        return df_clean
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            if option == 1:
                logging.debug("Option 1: Merging all DataFrames horizontally on one sheet.")
                all_dfs = list(itertools.chain.from_iterable(output_final))
                # Sanitize each DataFrame before concatenation
                all_dfs_clean = [sanitize_dataframe(df) for df in all_dfs]
                merged_df = pd.concat(all_dfs_clean, axis=0)
                merged_df.to_excel(writer, sheet_name=sanitize_worksheet_name("AllTablesMerged"), index=False)
                
            elif option == 2:
                logging.debug("Option 2: Each group on a different sheet, gap_rows between each.")
                for page_idx, df_group in enumerate(output_final):
                    sheet_name = sanitize_worksheet_name(f"Page_{page_idx+1}")
                    start_row = 0
                    for df in df_group:
                        # Sanitize the DataFrame
                        df_clean = sanitize_dataframe(df)
                        df_clean.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                        start_row += len(df_clean) + 1 + gap_rows
                        
            elif option == 3:
                logging.debug("Option 3: Flatten all DataFrames on one sheet vertically with gap_rows.")
                all_dfs = list(itertools.chain.from_iterable(output_final))
                sheet_name = sanitize_worksheet_name("AllTablesWithGaps")
                start_row = 0
                for df in all_dfs:
                    # Sanitize the DataFrame
                    df_clean = sanitize_dataframe(df)
                    df_clean.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                    start_row += len(df_clean) + 1 + gap_rows
                
            else:
                logging.error("Invalid `option` provided to write_output_final.")
                raise ValueError("Invalid `option` - must be 1, 2, or 3.")
    
    except Exception as e:
        logging.error(f"Error writing to Excel: {str(e)}")
        raise  # Re-raise the exception after logging

    logging.info("Excel file writing complete.")

def write_output_to_csv(output_final, csv_base_path, option=1, gap_rows=2):
    """
    Writes nested lists of DataFrames (`output_final`) to CSV files in 3 different ways.

    :param output_final: A list of lists of DataFrames. 
    :param csv_base_path: Base path/filename for CSV output (without extension)
    :param option: Choose 1 of 3 write modes:
                   1 = Horizontally merge all DataFrames into one CSV file
                   2 = Each top-level group in its own CSV file, with gap rows between tables
                   3 = Flatten all DataFrames into one CSV file with gap rows between them
    :param gap_rows: How many blank rows to insert between tables (for options 2 and 3).
    :return: List of paths to generated CSV files
    """
    logging.info(f"Writing output to CSV at '{csv_base_path}' with option={option}.")
    generated_files = []
    
    # Helper function to sanitize DataFrames before writing to CSV
    def sanitize_dataframe(df):
        df_clean = df.copy()
        
        # Replace problematic characters in column names
        df_clean.columns = [re.sub(r'[\[\]:*?/\\]', '_', str(col)) for col in df_clean.columns]
        
        # Replace problematic characters in string data
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # Only process string columns
                df_clean[col] = df_clean[col].astype(str).apply(
                    lambda x: re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', x) if pd.notna(x) else x
                )
        
        return df_clean
    
    try:
        if option == 1:
            logging.debug("Option 1: Merging all DataFrames into one CSV file.")
            all_dfs = list(itertools.chain.from_iterable(output_final))
            # Sanitize each DataFrame before concatenation
            all_dfs_clean = [sanitize_dataframe(df) for df in all_dfs]
            merged_df = pd.concat(all_dfs_clean, axis=0)
            
            csv_path = f"{csv_base_path}_concatenated.csv"
            merged_df.to_csv(csv_path, index=False)
            generated_files.append(csv_path)
            
        elif option == 2:
            logging.debug("Option 2: Each group in a separate CSV file.")
            for page_idx, df_group in enumerate(output_final):
                if not df_group:  # Skip empty groups
                    continue
                    
                # Create a new DataFrame for each page with appropriate gaps
                result_df = pd.DataFrame()
                current_row = 0
                
                for df in df_group:
                    df_clean = sanitize_dataframe(df)
                    
                    # Add blank rows if not at the start
                    if current_row > 0:
                        for _ in range(gap_rows):
                            result_df = pd.concat([result_df, pd.DataFrame([[''] * len(df_clean.columns)], columns=df_clean.columns)])
                            current_row += 1
                    
                    # Add the actual data
                    result_df = pd.concat([result_df, df_clean])
                    current_row += len(df_clean)
                
                csv_path = f"{csv_base_path}_page_{page_idx+1}.csv"
                result_df.to_csv(csv_path, index=False)
                generated_files.append(csv_path)
                
        elif option == 3:
            logging.debug("Option 3: Flatten all DataFrames into one CSV with gaps.")
            all_dfs = list(itertools.chain.from_iterable(output_final))
            
            # First determine the maximum column count across all tables
            max_cols = max([len(df.columns) for df in all_dfs]) if all_dfs else 0
            
            # Create a new large DataFrame with appropriate gaps
            result_df = pd.DataFrame()
            
            for i, df in enumerate(all_dfs):
                df_clean = sanitize_dataframe(df)
                
                # Add blank rows if not at the start
                if i > 0:
                    blank_df = pd.DataFrame([[''] * max_cols])
                    for _ in range(gap_rows):
                        result_df = pd.concat([result_df, blank_df])
                
                # Add the current DataFrame
                result_df = pd.concat([result_df, df_clean])
            
            csv_path = f"{csv_base_path}_all_tables_with_gaps.csv"
            result_df.to_csv(csv_path, index=False)
            generated_files.append(csv_path)
            
        else:
            logging.error("Invalid `option` provided to write_output_to_csv.")
            raise ValueError("Invalid `option` - must be 1, 2, or 3.")
    
    except Exception as e:
        logging.error(f"Error writing to CSV: {str(e)}")
        raise  # Re-raise the exception after logging

    logging.info(f"CSV file writing complete. Generated files: {generated_files}")
    return generated_files



