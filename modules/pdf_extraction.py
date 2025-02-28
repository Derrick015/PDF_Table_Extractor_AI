
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

def get_page_pixel_data(pdf_path, page_no, dpi=300, image_type='png'):
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
        tuple: (num_tables, table_headers, table_location)
    """
    logging.debug("Extracting table info from text.")

    match_num_tables = re.search(r'Number of Tables on the Page:\s*(\d+)', text)
    if match_num_tables:
        num_tables = int(match_num_tables.group(1))
        logging.debug(f"Found number of tables: {num_tables}")
    else:
        num_tables = None
        logging.debug("No table count found in text.")

    match_headers = re.search(r'Table Headers:\s*(.*?)\s*\n\s*3\.', text, re.DOTALL)
    if match_headers:
        headers_text = match_headers.group(1)
        table_headers = [h.strip() for h in headers_text.split('||')]
        logging.debug(f"Extracted table headers: {table_headers}")
    else:
        table_headers = []
        logging.debug("No table headers found.")

    match_location = re.search(r'Table Location for each table:\s*(.*)', text, re.DOTALL)
    if match_location:
        table_location = match_location.group(1).strip()
    else:
        table_location = ""

    table_location = table_location.split(" || ")
    logging.debug(f"Extracted table locations: {table_location}")

    return num_tables, table_headers, table_location

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

async def get_validated_table_info(text_input, open_api_key, base64_image, model='gpt-4o'):
    """
    Attempt to retrieve consistent table information by making multiple calls
    to the table identification LLM. If there's a majority match or exact match
    between attempts, return that; otherwise return the third attempt's output.
    """
    logging.info("Validating table information with multiple LLM calls.")

    async def asycn_pattern_desc():
        return await table_identification_llm(
            text_input=text_input,
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

    num_tables1, headers1, table_location = extract_table_info(output1)
    num_tables2, headers2, _ = extract_table_info(output2)


    if compare_table_headers(headers1, headers2) or (num_tables1 == num_tables2 and num_tables1 is not None):
        logging.info("Initial table info match or same table count. Returning first attempt's result.")
        return num_tables1, headers1, table_location, 0

    # Create third task if needed
    async with asyncio.TaskGroup() as tg:
        task3 = tg.create_task(asycn_pattern_desc())
    output3 = await task3
    logging.debug(f"LLM attempt 3 output:\n{output3}")

    num_tables3, headers3, _ = extract_table_info(output3)

    if compare_table_headers(headers3, headers1) or (num_tables3 == num_tables1 and num_tables3 is not None):
        logging.info("Majority match found with first and third results.")
        return num_tables1, headers1, table_location, 1
    
    if compare_table_headers(headers3, headers2) or (num_tables3 == num_tables2 and num_tables3 is not None):
        logging.info("Majority match found with second and third results.")
        return num_tables2, headers2, table_location, 1

    logging.warning("No matches found. Returning third run results for table_headers.")
    return num_tables3, headers3, table_location, 2

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
    table_location, 
    user_text, 
    extracted_text, 
    base64_image, 
    open_api_key, 
    page_number,
    max_retries=3,
    initial_delay=1,
    backoff_factor=2
):
    """
    Process tables by calling an LLM parser with exponential backoff.
    """
    logging.info(f"Processing tables to DataFrame for page {page_number + 1}")
    results_output = []

    # 1) Try first model: 'gpt-4o'
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            logging.debug(f"[Model gpt-4o] Attempt {attempt+1} of {max_retries}. Delay={delay}")
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
                            model='gpt-4o'
                        )
                    ))
            gpt4o_results = [await t for t in tasks]
            results_output = gpt4o_results
            logging.info("Successfully retrieved data using model 'gpt-4o'.")
            break
        except Exception as e:
            logging.warning(
                f"[Model gpt-4o] Attempt {attempt+1} of {max_retries} failed: {e}. "
                f"Retrying in {delay} second(s)..."
            )
            if attempt == max_retries - 1:
                logging.warning("Max retries with 'gpt-4o' exhausted; will try 'o1' next.")
            else:
                await asyncio.sleep(delay)
                delay *= backoff_factor

    # 2) If we got no results from 'gpt-4o', try 'o1'
    if not results_output:
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                logging.debug(f"[Model o1] Attempt {attempt+1} of {max_retries}. Delay={delay}")
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
                                model='o1'
                            )
                        ))
                o1_results = [await t for t in tasks]
                results_output = o1_results
                logging.info("Successfully retrieved data using model 'o1'.")
                break
            except Exception as e:
                logging.warning(
                    f"[Model o1] Attempt {attempt+1} of {max_retries} failed: {e}. "
                    f"Retrying in {delay} second(s)..."
                )
                if attempt == max_retries - 1:
                    logging.error("Processing failed - table could not be extracted after all retries.")
                    raise ValueError(f"Processing failed - table could not be extracted: {str(e)}")
                else:
                    await asyncio.sleep(delay)
                    delay *= backoff_factor

    # 3) Process the results into DataFrames
    df_list = []
    for i, out in enumerate(results_output):
        try:
            df = extract_df_from_string(out)
            logging.debug(f"Parsed DataFrame for table index {i} with shape {df.shape}")
        except:
            logging.warning(f"Could not extract table with index {i}, skipping.")
            continue

        # Normalize columns
        df.columns = df.columns.astype(str).str.strip().str.strip('"\'').str.title()
        if table_location[i] == 'Table is present in both the image and the text document':
            df[df.columns] = df[df.columns].map(
                lambda val: val if str(val) in extracted_text else "N/A"
            )
            df['table_header_descriptor'] = table_headers[i]
        else:
            df['table_header_descriptor'] = table_headers[i]
            df['page_number'] = page_number + 1

        df_list.append(df)

    logging.info(f"Completed processing tables to DataFrame for page {page_number + 1}.")
    return df_list

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
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        if option == 1:
            logging.debug("Option 1: Merging all DataFrames horizontally on one sheet.")
            all_dfs = list(itertools.chain.from_iterable(output_final))
            merged_df = pd.concat(all_dfs, axis=0)
            merged_df.to_excel(writer, sheet_name="AllTablesMerged", index=False)
            
        elif option == 2:
            logging.debug("Option 2: Each group on a different sheet, gap_rows between each.")
            for page_idx, df_group in enumerate(output_final):
                sheet_name = f"Page_{page_idx+1}"
                start_row = 0
                for df in df_group:
                    df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                    start_row += len(df) + 1 + gap_rows
                    
        elif option == 3:
            logging.debug("Option 3: Flatten all DataFrames on one sheet vertically with gap_rows.")
            all_dfs = list(itertools.chain.from_iterable(output_final))
            sheet_name = "AllTablesWithGaps"
            start_row = 0
            for df in all_dfs:
                df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
                start_row += len(df) + 1 + gap_rows
                
        else:
            logging.error("Invalid `option` provided to write_output_final.")
            raise ValueError("Invalid `option` - must be 1, 2, or 3.")

    logging.info("Excel file writing complete.")






# import pymupdf  
# import tkinter as tk
# from tkinter import filedialog
# import logging
# import pymupdf 
# import base64
# import pandas as pd
# import re
# import ast
# import itertools
# import asyncio
# from modules.llm import table_identification_llm, vision_column_llm_parser, vision_llm_parser
# import aiohttp


# # Configure logging: You can adjust the level to DEBUG for more detailed output.
# logging.basicConfig(
#     level=logging.INFO,  # Change to logging.DEBUG for more granular messages.
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )



# def extract_text_from_pages(pdf_input, pages=None):
#     """
#     Extracts text from specified pages in a PDF file using PyMuPDF.
    
#     Parameters:
#         pdf_input (str or file-like object): The path to the PDF file or a file-like object.
#         pages (int, list, tuple, or None): 
#             - If an integer, extracts text from that specific page (0-indexed).
#             - If a list of integers, extracts text from the specified pages.
#             - If a tuple of two integers, treats it as a range (start, end) and extracts from start (inclusive)
#               to end (exclusive).
#             - If None, extracts text from all pages.
    
#     Returns:
#         str: The concatenated text extracted from the specified pages.
#     """
#     text = ""
    
#     ## pages = pages-1 # adding this in as python starts from 0 put people don't

#     # Open the PDF file using PyMuPDF.  
#     # If pdf_input is a string, assume it's a file path.
#     if isinstance(pdf_input, str):
#         doc = pymupdf.open(pdf_input)
#     else:
#         # Otherwise, assume it's a file-like object (e.g., from Streamlit uploader)
#         # Make sure it's at the beginning (reset pointer) and get its bytes.
#         pdf_input.seek(0)
#         pdf_bytes = pdf_input.read()
#         doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    
#     total_pages = doc.page_count  # total number of pages in the document
    
#     # Determine which pages to extract.
#     if pages is None:
#         # Extract text from all pages.
#         page_indices = range(total_pages)
#     elif isinstance(pages, int):
#         if pages < 0 or pages >= total_pages:
#             raise ValueError(f"Page index {pages} is out of range. Total pages: {total_pages}")
#         page_indices = [pages]
#     elif isinstance(pages, (list, tuple)):
#         if isinstance(pages, tuple) and len(pages) == 2:
#             start, end = pages
#             if not (isinstance(start, int) and isinstance(end, int)):
#                 raise ValueError("Start and end values must be integers.")
#             if start < 0 or end > total_pages or start >= end:
#                 raise ValueError("Invalid page range specified.")
#             page_indices = range(start, end)
#         else:
#             page_indices = []
#             for p in pages:
#                 if not isinstance(p, int):
#                     raise ValueError("Page indices must be integers.")
#                 if p < 0 or p >= total_pages:
#                     raise ValueError(f"Page index {p} is out of range. Total pages: {total_pages}")
#                 page_indices.append(p)
#     else:
#         raise ValueError("Parameter 'pages' must be an int, list, tuple, or None.")
    
#     # Extract text from the specified pages.
#     for i in page_indices:
#         page = doc.load_page(i)
#         page_text = page.get_text()
#         text += f"\n\n--- Page {i + 1} ---\n\n" + page_text + "\n|-|+++|-|\n" # "\n|-|+++|-|\n" will be a delimiter ill use to seperate the pages where needed
    
#     doc.close()
#     return text



# def select_pdf_file():
#     """
#     Opens a file dialog for the user to select a PDF file.

#     Returns:
#         str: The path to the selected PDF file, or an empty string if no file was selected.
#     """
#     logging.info("Opening file selection dialog.")
#     # Create the Tkinter root window.
#     root = tk.Tk()
#     # Hide the main window.
#     root.withdraw()
#     # Set the root window to be topmost so the dialog appears in front.
#     root.attributes("-topmost", True)
    
#     # Open the file dialog for PDF files.
#     pdf_path = filedialog.askopenfilename(
#         title="Select a PDF file",
#         filetypes=[("PDF Files", "*.pdf")]
#     )
    
#     # Destroy the root window now that we're done with it.
#     root.destroy()
    
#     if pdf_path:
#         logging.info("Selected PDF file: %s", pdf_path)
#     else:
#         logging.info("No PDF file was selected.")
#     return pdf_path


# def get_page_pixel_data(pdf_path, page_no, dpi = 300, image_type = 'png'):
#     # Open the PDF file
#     doc = pymupdf.open(pdf_path)

#     # Select the first page (pages are 0-indexed)
#     page = doc[page_no]

#     # Calculate zoom factor for 300 DPI (since 300/72 ≈ 4.17)
#     zoom_factor = dpi / 72

#     matrix = pymupdf.Matrix(zoom_factor, zoom_factor)

#     # Render the page to an image (pixmap) using the transformation matrix
#     pix = page.get_pixmap(matrix=matrix)

#     png_data = pix.tobytes(image_type)

#     base64_image = base64.b64encode(png_data).decode('utf-8')

#     return base64_image


# def extract_table_info(text):
#     """
#     Extracts table information from a pattern description string.
    
#     Parameters:
#         pattern_desc_from_image (str): Text containing table pattern description
        
#     Returns:
#         tuple: Contains:
#             - num_tables (int): Number of tables found, or None if not found
#             - table_headers (list): List of table header strings
#             - parsing_rules (str): Rules for parsing tables and headers
#     """
    
#     # 1) Extract the Number of Tables
#     match_num_tables = re.search(r'Number of Tables on the Page:\s*(\d+)', text)
#     if match_num_tables:
#         num_tables = int(match_num_tables.group(1))
#     else:
#         num_tables = None  # or handle error

#     # 2) Extract the Table Headers
#     #    We'll grab everything after "Table Headers:" until the line "3. Rules" begins
#     match_headers = re.search(r'Table Headers:\s*(.*?)\s*\n\s*3\.', text, re.DOTALL)
#     if match_headers:
#         headers_text = match_headers.group(1)
#         # Split on '||' to get individual headers
#         table_headers = [h.strip() for h in headers_text.split('||')]
#     else:
#         table_headers = []


#     # 3) Extract the Table Location
#     match_location = re.search(r'Table Location for each table:\s*(.*)', text, re.DOTALL)
#     if match_location:
#         table_location = match_location.group(1).strip()
#     else:
#         table_location = ""

#     table_location = table_location.split(" || ")

#     return num_tables, table_headers, table_location


# def compare_table_headers(headers1, headers2):
#     """
#     Compare two lists of table headers and return True if they are the same.
#     """
#     if len(headers1) != len(headers2):
#         return False
    
#     return all(h1.strip() == h2.strip() for h1, h2 in zip(headers1, headers2))



# async def get_validated_table_info(text_input, open_api_key, base64_image, model='gpt-4o'):

#     async def asycn_pattern_desc():

#         pattern_desc = await table_identification_llm(
#             text_input=text_input,
#             base64_image=base64_image,
#             open_api_key=open_api_key,
#             model=model
#         )
#         return pattern_desc


#     tasks = []
#     # Create all tasks first
#     async with asyncio.TaskGroup() as tg:
#         tasks.append(tg.create_task(asycn_pattern_desc()))
#         tasks.append(tg.create_task(asycn_pattern_desc()))
    
#     # Get results after tasks complete
#     output1 = await tasks[0]
#     output2 = await tasks[1]
    
#     num_tables1, headers1, table_location = extract_table_info(output1)
#     num_tables2, headers2, table_location = extract_table_info(output2)
#     print(num_tables1, num_tables2)
#     if compare_table_headers(headers1, headers2) or (num_tables1 == num_tables2 and num_tables1 is not None):
#         print('Initial headers match or same number of tables')
#         return num_tables1, headers1, table_location, 0

#     # Create third task if needed
#     async with asyncio.TaskGroup() as tg:
#         task3 = tg.create_task(asycn_pattern_desc())
    
#     output3 = await task3
#     num_tables3, headers3, table_location = extract_table_info(output3)
    
#     # If headers match exactly or number of tables is the same, use first run results
#     if compare_table_headers(headers3, headers1) or (num_tables3 == num_tables1 and num_tables3 is not None):
#         print('Majority match found with first and third results')
#         return num_tables1, headers1, table_location, 1
    
#     # If headers match exactly or number of tables is the same, use first run results
#     if compare_table_headers(headers3, headers2) or (num_tables3 == num_tables2 and num_tables3 is not None):
#         print('Majority match found with second and third results')
#         return num_tables2, headers2, table_location, 1

#     # If no matches found, return the third run results 
#     print('No matches found. Returning third run results for table_headers')
#     return num_tables3, headers3, table_location, 2



# def compare_column_data(data1, data2):
#     issue_table_headers = []
#     """
#     Compare two sets of column data results and return True if they match.
#     Compares column names for each index to ensure consistency.
#     """
#     if len(data1) != len(data2):
#         logging.warning("Column data length mismatch")
#         return False
    
#     # Sort both lists by index to ensure we compare corresponding entries
#     data1_sorted = sorted(data1, key=lambda x: x["index"])
#     data2_sorted = sorted(data2, key=lambda x: x["index"])
    
#     for item1, item2 in zip(data1_sorted, data2_sorted):
#         # Compare column names for the same index (order doesn't matter)
#         if set(item1["column_names"]) != set(item2["column_names"]):
#             logging.warning(f"Column names mismatch for index {item1['index']}")
#             logging.warning(f"Set 1: {item1['column_names']}")
#             logging.warning(f"Set 2: {item2['column_names']}")
#             issue_table_headers.append(item1["table_header"])
#             return False, issue_table_headers
            
#     return True, issue_table_headers


# def extract_columns(response_text, tables_to_target):
#     pattern = r'index:\s*\[(\d+)\].*?column_names:\s*\[(.*?)\].*?example_value_per_column:\s*\[(.*?)\].*?table_location:\s*\[(.*?)\]'
#     matches = re.findall(pattern, response_text)
    
#     results = []
#     for index_str, columns_str, example_values_str, location_str in matches:
#         index_value = int(index_str)
#         columns_list = [col.strip().strip('"\'') for col in columns_str.split(',')]
        
#         # Parse example values into a dictionary
#         example_values = {}
#         for pair in example_values_str.split(','):
#             if ':' in pair:
#                 key, value = pair.split(':')
#                 key = key.strip().strip('"\'')
#                 value = value.strip().strip('()').strip('"\'')
#                 example_values[key] = value
        
#         header = tables_to_target[index_value]
        
#         results.append({
#             "index": index_value,
#             "table_header": header,
#             "column_names": columns_list,
#             "example_values_per_column": example_values,
#             "table_location": location_str.strip()
#         })
#     return results


# async def parse_column_data(user_text, text_input, tables_to_target, base64_image, table_list, open_api_key, model='gpt-4o'):
#     async def async_pattern_desc():
#         max_retries = 3
#         retry_delay = 1
        
#         for attempt in range(max_retries):
#             try:
#                 llm_parser_desc = await vision_column_llm_parser(
#                     user_text=user_text,
#                     text_input=text_input,
#                     table_to_target=tables_to_target,
#                     base64_image=base64_image,
#                     open_api_key=open_api_key,
#                     model=model
#                 )
#                 return llm_parser_desc
#             except Exception as e:
#                 if attempt == max_retries - 1:  # Last attempt
#                     raise  # Re-raise the last exception
#                 logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
#                 await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff

#     tasks = []
#     # Create all tasks first
#     try:
#         async with asyncio.TaskGroup() as tg:
#             tasks.append(tg.create_task(async_pattern_desc()))
#             tasks.append(tg.create_task(async_pattern_desc()))

#         # Get results after tasks complete
#         output1 = await tasks[0]
#         output2 = await tasks[1]

#         results1 = extract_columns(output1, table_list)
#         results2 = extract_columns(output2, table_list)

#         bool_decision, issue_table_headers_0 = compare_column_data(results1, results2)
        
#         if bool_decision:
#             logging.info("Initial phase column parsing match")
#             return results1, issue_table_headers_0, 0  # Return score 0 for perfect match
        
#         # Create third task if needed
#         async with asyncio.TaskGroup() as tg:
#             task3 = tg.create_task(async_pattern_desc())
        
#         output3 = await task3
#         results3 = extract_columns(output3, table_list)

#         logging.warning(f"\n comparing results 1 and 3")
#         bool_decision, issue_table_headers = compare_column_data(results1, results3)
#         if bool_decision:
#             logging.warning("Majority match found with first and third results")
#             return results1, issue_table_headers_0, 1 # If the the majority matches if found then there will not be an issue table headers thus return the initial one that caused the issue in the first place. 
        
#         logging.warning(f"\n comparing results 2 and 3")
#         bool_decision, issue_table_headers = compare_column_data(results2, results3)
#         if bool_decision:
#             logging.warning("Majority match found with second and third results")
#             return results2, issue_table_headers_0, 1
        
#         # If no majority found, return third result with low confidence score
#         logging.warning("No matching results found, returning third result with low confidence")
#         return results3, issue_table_headers, 2

#     except aiohttp.ClientError as e:
#         logging.error(f"Network error occurred: {str(e)}")
#         raise
#     except Exception as e:
#         logging.error(f"Unexpected error: {str(e)}")
#         raise



# def parse_variable_data_to_df(text):
#     """
#     Parse variable data text into a pandas DataFrame.
    
#     Args:
#         text (str): Text containing variable data in format [column:val1 |-| val2 |-| ...]
        
#     Returns:
#         pd.DataFrame: DataFrame containing the parsed data with columns padded to equal length
#     """
#     pattern = r"\[([^\]:]+):([^]]+)\]"
#     matches = re.findall(pattern, text, flags=re.DOTALL)

#     data = {}
#     max_len = 0

#     # Split each matched value on "|-|"
#     for key, val in matches:
#         items = [item.replace("***", "").strip()  # optionally remove "***"
#                  for item in val.split("|-|")]
#         data[key.strip()] = items
#         max_len = max(max_len, len(items))

#     # Build a DataFrame, ensuring all columns have the same length
#     df = pd.DataFrame({
#         col: values + [None]*(max_len - len(values))  # pad with None if needed
#         for col, values in data.items()
#     })

#     return df


# def extract_df_from_string(text):
#     # Find the content between the first [ and last ], including the brackets
#     match = re.search(r'(\[.*\])', text, re.DOTALL)
#     if match:
#         # Convert the matched string to a Python object using ast.literal_eval
#         data = ast.literal_eval(match.group(1))
#         # Create DataFrame from the data
#         df = pd.DataFrame(data)
#         return df
#     raise ValueError("No tables extracted from the page")




# async def process_tables_to_df(
#     table_headers, 
#     table_location, 
#     user_text, 
#     extracted_text, 
#     base64_image, 
#     open_api_key, 
#     page_number,
#     max_retries=3,            # Maximum number of retries
#     initial_delay=1,          # Initial delay in seconds
#     backoff_factor=2          # Multiplicative factor for each backoff
# ):
#     """
#     Process tables by calling an LLM parser with exponential backoff.

#     :param table_headers: List of table header descriptors
#     :param table_location: List with table location descriptions
#     :param user_text: The user prompt
#     :param extracted_text: The text extracted from the document
#     :param base64_image: The base64-encoded image
#     :param open_api_key: API key for the LLM
#     :param page_number: The page number (for labeling)
#     :param max_retries: How many times to retry (per model) before giving up
#     :param initial_delay: How many seconds to wait before the first retry
#     :param backoff_factor: Multiply the delay by this after each failed attempt
#     """
#     results_output = []

#     # --------------------------------------------------------------
#     # 1) Try first model: 'gpt-4o' with exponential backoff
#     # --------------------------------------------------------------
#     delay = initial_delay  # current delay used for the exponential backoff
#     for attempt in range(max_retries):
#         try:
#             # Create and run all tasks concurrently
#             tasks = []
#             async with asyncio.TaskGroup() as tg:
#                 for table in table_headers:
#                     tasks.append(tg.create_task(
#                         vision_llm_parser(
#                             user_text=user_text,
#                             text_input=extracted_text,
#                             table_to_target=table,
#                             base64_image=base64_image,
#                             open_api_key=open_api_key,
#                             model='gpt-4o'
#                         )
#                     ))
#             # Gather results
#             gpt4o_results = []
#             for task in tasks:
#                 gpt4o_results.append(await task)

#             # If we get here without an exception, it means success:
#             results_output = gpt4o_results
#             break  # break out of the for-loop; no more retries needed

#         except Exception as e:
#             logging.warning(
#                 f"[Model gpt-4o] Attempt {attempt+1} of {max_retries} failed: {e}. "
#                 f"Retrying in {delay} second(s)..."
#             )
#             if attempt == max_retries - 1:
#                 logging.warning("Max retries with 'gpt-4o' exhausted; will try 'o1' next.")
#             else:
#                 # Wait for the current delay, then increase it
#                 await asyncio.sleep(delay)
#                 delay *= backoff_factor

#     # --------------------------------------------------------------
#     # 2) If we got no results from 'gpt-4o', try model: 'o1'
#     # --------------------------------------------------------------
#     if not results_output:
#         delay = initial_delay  # reset delay for new model attempts
#         for attempt in range(max_retries):
#             try:
#                 tasks = []
#                 async with asyncio.TaskGroup() as tg:
#                     for table in table_headers:
#                         tasks.append(tg.create_task(
#                             vision_llm_parser(
#                                 user_text=user_text,
#                                 text_input=extracted_text,
#                                 table_to_target=table,
#                                 base64_image=base64_image,
#                                 open_api_key=open_api_key,
#                                 model='o1'
#                             )
#                         ))
#                 o1_results = []
#                 for task in tasks:
#                     o1_results.append(await task)

#                 results_output = o1_results
#                 break

#             except Exception as e:
#                 logging.warning(
#                     f"[Model o1] Attempt {attempt+1} of {max_retries} failed: {e}. "
#                     f"Retrying in {delay} second(s)..."
#                 )
#                 if attempt == max_retries - 1:
#                     # No more retries left, raise final error
#                     raise ValueError(f"Processing failed - table could not be extracted: {str(e)}")
#                 else:
#                     await asyncio.sleep(delay)
#                     delay *= backoff_factor

#     # --------------------------------------------------------------
#     # 3) Process the results into DataFrames
#     # --------------------------------------------------------------
#     df_list = []
#     for i, out in enumerate(results_output):
#         try:
#             df = extract_df_from_string(out)
#         except:
#             logging.warning(f"Could not extract table with index {i}, skipping")
#             continue

#         # Normalize columns
#         df.columns = df.columns.astype(str).str.strip().str.strip('"\'').str.title()

#         if table_location[i] == 'Table is present in both the image and the text document':
#             # If val is not in extracted_text, mark it as "N/A"
#             df[df.columns] = df[df.columns].map(
#                 lambda val: val if str(val) in extracted_text else "N/A"
#             )
#             df['table_header_descriptor'] = table_headers[i]
#         else:
#             df['table_header_descriptor'] = table_headers[i]

#             df['page_number'] = page_number + 1
            
#             df_list.append(df)
            

            


#     return df_list


# def write_output_final(output_final, excel_path, option=1, gap_rows=2):
#     """
#     Writes nested lists of DataFrames (`output_final`) to Excel in 3 different ways.

#     :param output_final: A list of lists of DataFrames. 
#                         e.g. [
#                             [df0_0, df0_1, df0_2],  # first "page"
#                             [df1_0, df1_1],        # second "page"
#                             ...
#                         ]
#     :param excel_path: Output Excel filename/path
#     :param option: Choose 1 of 3 write modes:
#                    1 = Horizontally merge (side-by-side) all DataFrames into one wide table (one sheet)
#                    2 = Each top-level group on its own sheet, with `gap_rows` blank rows between sub-DataFrames
#                    3 = Flatten all DataFrames onto one sheet vertically, with `gap_rows` blank rows between them
#     :param gap_rows: How many blank rows to insert between tables (used in options 2 and 3).
#     """
#     with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
#         if option == 1:
#             # ---------------------------------------------------------
#             # (1) Horizontally merge *all* DataFrames into ONE wide table
#             # ---------------------------------------------------------
#             # Flatten the nested list into a single list of DataFrames
#             all_dfs = list(itertools.chain.from_iterable(output_final))
            
#             # Concatenate them horizontally (side-by-side)
#             # By default, rows are aligned on the index; 
#             # if DataFrames have different row indices, you may need join='outer' or join='inner'
#             merged_df = pd.concat(all_dfs, axis=0)
#             merged_df.to_excel(writer, sheet_name="AllTablesMerged", index=False)
            
#         elif option == 2:
#             # ---------------------------------------------------------
#             # (2) Each top-level group on a DIFFERENT sheet,
#             #     with gap_rows between each sub-DataFrame
#             # ---------------------------------------------------------
#             for page_idx, df_group in enumerate(output_final):
#                 # Each top-level group is a "page" => one sheet
#                 sheet_name = f"Page_{page_idx+1}"
#                 start_row = 0
                
#                 for df in df_group:
#                     df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
#                     # Shift down for the next DF: table rows + header row + gap_rows
#                     start_row += len(df) + 1 + gap_rows
                    
#         elif option == 3:
#             # ---------------------------------------------------------
#             # (3) Flatten all DataFrames on ONE sheet (stacked vertically),
#             #     with gap_rows rows between each
#             # ---------------------------------------------------------
            
#             all_dfs = list(itertools.chain.from_iterable(output_final))
            
#             sheet_name = "AllTablesWithGaps"
#             start_row = 0
#             for df in all_dfs:
#                 df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
#                 start_row += len(df) + 1 + gap_rows
                
#         else:
#             raise ValueError("Invalid `option` - must be 1, 2, or 3.")



