# PDF Table Extractor AI

A powerful tool that uses AI vision models to extract tabular data from PDF documents.

## Overview

PDF Table Extractor AI is designed to solve the challenging problem of extracting structured table data from PDF files. It leverages OpenAI's advanced vision models to identify, parse, and extract tables, even from complex PDFs with varying layouts and formats.

## Key Features

- **AI-Powered Table Detection**: Automatically identifies tables within PDF documents
- **Data Extraction**: Preserves table structure and relationships between data elements
- **Multiple Page Processing**: Process specific pages, page ranges, or entire documents
- **Customizable AI Instructions**: Provide specific extraction instructions for tailored results
- **Multiple Export Formats**: Download extracted tables in various Excel formats
- **User-Friendly Interface**: Simple web-based UI built with Streamlit

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key 

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/Derrick015/PDF_Table_Extractor_AI.git
   cd PDF_Table_Extractor_AI
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Running the Application

Start the Streamlit web application:

```
streamlit run app.py
```

The application will be available at http://localhost:8501 in your web browser.

### Extracting Tables

1. Upload a PDF file using the file uploader
2. Select pages to process (all pages, specific range, or custom selection)
3. Review page previews to confirm your selection
4. Optionally, customize the AI instructions in the sidebar
5. Click "Process Selected Pages" to start extraction
6. Download the results in your preferred format:
   - Format 1: All tables concatenated on a single sheet
   - Format 2: All tables on a page per sheet
   - Format 3: All tables on one sheet with spacing
   - Download All Formats (ZIP): Contains all three Excel formats in a single ZIP file

## How It Works

1. **Table Detection**: The AI vision model analyzes the PDF to identify and locate tables
2. **Structure Recognition**: The system determines table headers, rows, and columns
3. **Data Extraction**: Content is extracted while preserving the table structure
4. **Validation**: Multiple AI passes to mitigate hallucination and improve accuracy and consistency. Generated values are also cross checked with values on the PDF text and replaced with N/A if abscent. 
5. **Export**: Data is formatted into Excel spreadsheets for easy use

## Technical Details

- **PDF Processing**: Uses PyMuPDF for efficient PDF handling
- **AI Vision**: Leverages OpenAI's GPT-4o model for visual recognition
- **Concurrent Processing**: Implements asyncio for parallel page processing
- **Data Handling**: Pandas for structured data manipulation
- **Web Interface**: Streamlit for an intuitive user experience

## Limitations

- Performance depends on the quality and complexity of the PDF
- Processing large documents may take time and consume API credits
- Very complex or highly stylised tables may require manual verification
- Does not currenlty work with tables in images
- 200 MB max file size

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Derrick Owusu Ofori

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgements

- OpenAI for providing the vision models
- PyMuPDF for PDF processing capabilities
- Streamlit for the web application framework

## Contact

For questions, support, or collaboration:

- GitHub Issues: Please use the [issue tracker](https://github.com/Derrick015/PDF_Table_Extractor_AI) for bug reports and feature requests
- Email: derrickowusuof@gmail.com
- LinkedIn: [Derrick Ofori](https://www.linkedin.com/in/derrickofori/)

Feel free to reach out through any of these channels. For bug reports and feature requests, please use GitHub Issues as the primary channel. 