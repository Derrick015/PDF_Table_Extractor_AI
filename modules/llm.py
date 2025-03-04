

from pydantic import BaseModel
import multiprocessing
import aiohttp
import requests

num_cpus = multiprocessing.cpu_count()


# Strucuted Class

# 1. Variable_extractor_llm
class structed_output_class_variable_extractor(BaseModel):
    output: list[str]




def variable_extractor_llm(user_text, text_data, openai_client, model = "gpt-4o-mini", response_format = structed_output_class_variable_extractor):

  completion = openai_client.beta.chat.completions.parse(
      model= model,
      messages=[
          {"role": "system", "content": """ 
          
        You are a skilled data extraction and parsing specialist with expertise in analysing PDF text documents and extracting structured information. 
        Your job is to read a user's request and the document being refered to and identify the variable(s)/column(s) the user wishes to extract and return them in a list. 

        For a giving variable/column ensure the spaces within it are replaced with an under score for example (if the variable is "pack quantity" ensure you return "pack_qunatity") and ensure it is lower cased


        Example
        
        ``````````````````````
        User's request: I want to extract all data in the columns as well as the header which is ontop of the table
        ``````````````````````

        ``````````````````````
        Text document: '\n\n--- Page 1 ---\n\n21\n1.01  Hydraulic Hose & Inserts\nHydraulics\n1\n Metric Heavy Series\n Hexavalent chromium free plating\nMetric Male 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMMS03-M16CF\nM16 x 1.5\n3/16”\n8S\nHMMS04-M14CF\nM14 x 1.5\n1/4”\n6S\nHMMS04-M16CF\nM16 x 1.5\n1/4”\n8S\nHMMS04-M18CF\nM18 x 1.5\n1/4”\n10S\nHMMS05-M18CF\nM18 x 1.5\n5/16”\n10S\nHMMS05-M20CF\nM20 x 1.5\n5/16”\n12S\nHMMS06-M18CF\nM18 x 1.5\n3/8”\n10S\nHMMS06-M20CF\nM20 x 1.5\n3/8”\n12S\nHMMS06-M22CF\nM22 x 1.5\n3/8”\n14S\nHMMS06-M24CF\nM24 x 1.5\n3/8”\n16S\nHMMS08-M24CF\nM24 x 1.5\n1/2”\n16S\nHMMS10-M30CF\nM30 x 2\n5/8”\n20S\nHMMS12-M30CF\nM30 x 2\n3/4”\n20S\nHMMS12-M36CF\nM36 x 2\n3/4”\n25S\nHMMS16-M36CF\nM36 x 1.5\n1”\n25S\nHMMS16-M42CF\nM42 x 2\n1”\n30S\nHMMS20-M42CF\nM42 x 2\n1.1/4”\n30S\nHMMS20-M52CF\nM52 x 2\n1.1/4”\n38S\nHMMS24-M52CF\nM52 x 2\n1.1/2”\n38S\nMetric Female 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMFS03-M16CF\nM16 x 1.5\n3/16”\n8S\nHMFS04-M14CF\nM14 x 1.5\n1/4”\n6S\nHMFS04-M16CF\nM16 x 1.5\n1/4”\n8S\nHMFS04-M18CF\nM18 x 1.5\n1/4”\n10S\nHMFS05-M18CF\nM18 x 1.5\n5/16”\n10S\nHMFS05-M20CF\nM20 x 1.5\n5/16”\n12S\nHMFS06-M16CF\nM16 x 1.5\n3/8”\n8S\nHMFS06-M18CF\nM18 x 1.5\n3/8”\n10S\nHMFS06-M20CF\nM20 x 1.5\n3/8”\n12S\nHMFS06-M22CF\nM22 x 1.5\n3/8”\n14S\nHMFS08-M22CF\nM22 x 1.5\n1/2”\n14S\nHMFS08-M24CF\nM24 x 1.5\n1/2”\n16S\nHMFS08-M30CF\nM30 x 2\n1/2”\n20S\nHMFS10-M24CF\nM24 x 1.5\n5/8”\n16S\nHMFS10-M30CF\nM30 x 2\n5/8”\n20S\nHMFS12-M30CF\nM30 x 2\n3/4”\n20S\nHMFS12-M36CF\nM36 x 2\n3/4”\n25S\nHMFS16-M36CF\nM36 x 2\n1”\n25S\nHMFS16-M42CF\nM42 x 2\n1”\n30S\nHMFS20-M52CF\nM52 x 2\n1.1/4”\n38S\nHMFS24-M52CF\nM52 x 2\n1.1/2”\n38S\n90¡ Metric Female 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMFS0390-M16CF\nM16 x 1.5\n3/16”\n8S\nHMFS0490-M14CF\nM14 x 1.5\n1/4”\n6S\nHMFS0490-M16CF\nM16 x 1.5\n1/4”\n8S\nHMFS0490-M18CF\nM18 x 1.5\n1/4”\n10S\nHMFS0490-M20CF\nM20 x 1.5\n1/4”\n12S\nHMFS0590-M18CF\nM18 x 1.5\n5/16”\n10S\nHMFS0590-M20CF\nM20 x 1.5\n5/16”\n12S\nHMFS0690-M18CF\nM18 x 1.5\n3/8”\n10S\nHMFS0690-M20CF\nM20 x 1.5\n3/8”\n12S\nHMFS0690-M22CF\nM22 x 1.5\n3/8”\n14S\nHMFS0890-M24CF\nM24 x 1.5\n1/2”\n16S\nHMFS1090-M24CF\nM24 x 1.5\n5/8”\n16S\nHMFS1090-M30CF\nM30 x 2\n5/8”\n20S\nHMFS1290-M30CF\nM30 x 2\n3/4”\n20S\nHMFS1290-M36CF\nM36 x 2\n3/4”\n25S\nHMFS1690-M36CF\nM36 x 2\n1”\n25S\nHMFS1690-M42CF\nM42 x 2\n1”\n30S\nHMFS2090-M52CF\nM52 x 2\n1.1/4”\n38S\nHMFS2490-M52CF\nM52 x 2\n1.1/2”\n38S\nStandpipe, \nS Series\n0309\nOD mm\nTo Suit Hose ID\nHSSP04-10CF\n10\n1/4”\nHSSP06-14CF\n14\n3/8”\nHSSP08-16CF\n16\n1/2”\nHSSP10-20CF\n20\n5/8”\nHSSP12-20CF\n20\n3/4”\nHSSP12-25CF\n25\n3/4”\nHSSP16-25CF\n25\n1”\nHSSP16-30CF\n30\n1”\nHSSP20-38CF\n38\n1.1/4”\n45¡ Metric Female 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMFS03-45-M16CF\nM16 x 1.5\n3/16”\n8S\nHMFS04-45-M14CF\nM14 x 1.5\n1/4”\n6S\nHMFS04-45-M16CF\nM16 x 1.5\n1/4”\n8S\nHMFS04-45-M18CF\nM18 x 1.5\n1/4”\n10S\nHMFS05-45-M18CF\nM18 x 1.5\n5/16”\n10S\nHMFS05-45-M20CF\nM20 x 1.5\n5/16”\n12S\nHMFS06-45-M20CF\nM20 x 1.5\n3/8”\n12S\nHMFS06-45-M22CF\nM22 x 1.5\n3/8”\n14S\nHMFS08-45-M24CF\nM24 x 1.5\n1/2”\n16S\nHMFS10-45-M30CF\nM30 x 2\n5/8”\n20S\nHMFS12-45-M30CF\nM30 x 2\n3/4”\n20S\nHMFS12-45-M36CF\nM36 x 2\n3/4”\n25S\nHMFS16-45-M36CF\nM36 x 2\n1”\n25S\nHMFS16-45-M42CF\nM42 x 2\n1”\n30S\nHMFS20-45-M52CF\nM52 x 2\n1.1/4”\n38S\nHMFS24-45-M52CF\nM52 x 2\n1.1/2”\n38S\n90¡ Standpipe, \nS Series\n0309\nOD mm\nTo Suit Hose ID\nHSSP04-90-08CF\n8\n1/4”\nHSSP04-90-10CF\n10\n1/4”\nHSSP06-90-14CF\n14\n3/8”\nHSSP08-90-16CF\n16\n1/2”\nHSSP10-90-20CF\n20\n5/8”\nHSSP12-90-20CF\n20\n3/4”\nHSSP12-90-25CF\n25\n3/4”\nHSSP16-90-25CF\n25\n1”\nHSSP16-90-30CF\n30\n1”\n45¡ Standpipe, S Series\n0309\nOD mm\nTo Suit Hose ID\nHSSP04-45-08CF\n8\n1/4”\nHSSP04-45-10CF\n10\n1/4”\nHSSP04-45-12CF\n12\n1/4”\nHSSP05-45-12CF\n12\n5/16”\nHSSP06-45-14CF\n14\n3/8”\nHSSP08-45-14CF\n14\n1/2”\nHSSP08-45-16CF\n16\n1/2”\nHSSP10-45-16CF\n16\n5/8”\nHSSP10-45-20CF\n20\n5/8”\nHSSP12-45-20CF\n20\n3/4”\nHSSP12-45-25CF\n25\n3/4”\nHSSP16-45-25CF\n25\n1”\nHSSP16-45-30CF\n30\n1”\nHSSP16-45-38CF\n38\n1”\nHSSP20-45-30CF\n30\n1.1/4”\nHSSP20-45-38CF\n38\n1.1/4”\nHYDRAULIC HOSE CONNECTORS'
        ``````````````````````
      
        ``````````````````````                
        Output: [0309, thread, to_suit_house_id, od_mm, size, header]
        ``````````````````````  

           
           """},
          {"role": "user", "content": f"""

        Now provide the output for
        

        ``````````````````````  
        User's request: {user_text}
        ``````````````````````

        ``````````````````````      
        Text document: {text_data}
        ``````````````````````
          """},
      ],
      response_format = response_format
      #, temperature= 0.4z


)
  return completion.choices[0].message.parsed.output



def table_extraction(pattern_data, text_data, openai_client, model = "gpt-4o-mini", response_format = structed_output_class_variable_extractor):

  completion = openai_client.beta.chat.completions.parse(
      model= model,
      messages=[
          {"role": "system", "content": """ 
          
           
        You are a skilled data extraction and parsing specialist with expertise in analyzing PDF text documents and extracting structured information.
        You will be provided with textual information extracted from a PDF page, along with input from an expert who has analyzed both the image and text of the document.
        The expert will describe how tables, headings, and sections are laid out, providing a pattern to assist you in your task.
        
        You task is to extract a table header for each table on the page.
        Where a header for the table is absent provided a unified descriptor of the table based on its columns or other information in or around the table that distinguishes it. 

        Example
           
        ``````````````````````
        Description of layout of table, header, sections and patterns: 'The provided text from the PDF page contains two main sections with tables:\n\n### Section 1: Steel Enclosed Contactors\n- **Structure and Headers:**\n  - Headers: "Order No.", "Model No.", "AC1 Heating & General Load", "Fluorescent", "Mercury", "Halogen", "LED", "No. of Poles", "Dimensions HxWxD"\n  - Each row lists specifications for steel-enclosed contactors, such as order and model numbers, electrical ratings (AC1, Fluorescent, etc.), the number of poles, and dimensions.\n\n### Section 2: Direct On Line, Reversing & Star Delta Starters\n- **Structure and Headers:**\n  - Headers: "Order No.", "Model No.", "Max Rating", "Control Voltage", "Dimensions H x W x D", "Type"\n  - Each row provides details on different types of starters, including order and model numbers, maximum rating, control voltage, dimensions, and type.\n\n### Parsing Pattern:\n- **Patterns to Extract:**\n  - Identify sections by headers that are distinct and capitalized or emphasized (e.g., "STEEL ENCLOSED CONTACTORS").\n  - Each header is followed by corresponding data in a structured row format.\n  - New entries often follow a pattern of numbers followed by letters (e.g., order numbers).\n  - Delimiters between rows are spaces or newline characters.\n\n### Grouping:\n- Each section\'s data is grouped under its title and is separated by bullet points or non-table descriptive text.\n- Recognize transitions between sections by change in headers or the start of bullet points.\n\nBased on the image, sections are distinct and outlined clearly. The headers are bold and in larger font, followed by tabulated, organized data. These visual cues correspond with text organization, ensuring accurate text extraction and data parsing.'
        ``````````````````````

        ``````````````````````
        Text document: '\n\n--- Page 1 ---\n\n21\n1.01  Hydraulic Hose & Inserts\nHydraulics\n1\n Metric Heavy Series\n Hexavalent chromium free plating\nMetric Male 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMMS03-M16CF\nM16 x 1.5\n3/16”\n8S\nHMMS04-M14CF\nM14 x 1.5\n1/4”\n6S\nHMMS04-M16CF\nM16 x 1.5\n1/4”\n8S\nHMMS04-M18CF\nM18 x 1.5\n1/4”\n10S\nHMMS05-M18CF\nM18 x 1.5\n5/16”\n10S\nHMMS05-M20CF\nM20 x 1.5\n5/16”\n12S\nHMMS06-M18CF\nM18 x 1.5\n3/8”\n10S\nHMMS06-M20CF\nM20 x 1.5\n3/8”\n12S\nHMMS06-M22CF\nM22 x 1.5\n3/8”\n14S\nHMMS06-M24CF\nM24 x 1.5\n3/8”\n16S\nHMMS08-M24CF\nM24 x 1.5\n1/2”\n16S\nHMMS10-M30CF\nM30 x 2\n5/8”\n20S\nHMMS12-M30CF\nM30 x 2\n3/4”\n20S\nHMMS12-M36CF\nM36 x 2\n3/4”\n25S\nHMMS16-M36CF\nM36 x 1.5\n1”\n25S\nHMMS16-M42CF\nM42 x 2\n1”\n30S\nHMMS20-M42CF\nM42 x 2\n1.1/4”\n30S\nHMMS20-M52CF\nM52 x 2\n1.1/4”\n38S\nHMMS24-M52CF\nM52 x 2\n1.1/2”\n38S\nMetric Female 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMFS03-M16CF\nM16 x 1.5\n3/16”\n8S\nHMFS04-M14CF\nM14 x 1.5\n1/4”\n6S\nHMFS04-M16CF\nM16 x 1.5\n1/4”\n8S\nHMFS04-M18CF\nM18 x 1.5\n1/4”\n10S\nHMFS05-M18CF\nM18 x 1.5\n5/16”\n10S\nHMFS05-M20CF\nM20 x 1.5\n5/16”\n12S\nHMFS06-M16CF\nM16 x 1.5\n3/8”\n8S\nHMFS06-M18CF\nM18 x 1.5\n3/8”\n10S\nHMFS06-M20CF\nM20 x 1.5\n3/8”\n12S\nHMFS06-M22CF\nM22 x 1.5\n3/8”\n14S\nHMFS08-M22CF\nM22 x 1.5\n1/2”\n14S\nHMFS08-M24CF\nM24 x 1.5\n1/2”\n16S\nHMFS08-M30CF\nM30 x 2\n1/2”\n20S\nHMFS10-M24CF\nM24 x 1.5\n5/8”\n16S\nHMFS10-M30CF\nM30 x 2\n5/8”\n20S\nHMFS12-M30CF\nM30 x 2\n3/4”\n20S\nHMFS12-M36CF\nM36 x 2\n3/4”\n25S\nHMFS16-M36CF\nM36 x 2\n1”\n25S\nHMFS16-M42CF\nM42 x 2\n1”\n30S\nHMFS20-M52CF\nM52 x 2\n1.1/4”\n38S\nHMFS24-M52CF\nM52 x 2\n1.1/2”\n38S\n90¡ Metric Female 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMFS0390-M16CF\nM16 x 1.5\n3/16”\n8S\nHMFS0490-M14CF\nM14 x 1.5\n1/4”\n6S\nHMFS0490-M16CF\nM16 x 1.5\n1/4”\n8S\nHMFS0490-M18CF\nM18 x 1.5\n1/4”\n10S\nHMFS0490-M20CF\nM20 x 1.5\n1/4”\n12S\nHMFS0590-M18CF\nM18 x 1.5\n5/16”\n10S\nHMFS0590-M20CF\nM20 x 1.5\n5/16”\n12S\nHMFS0690-M18CF\nM18 x 1.5\n3/8”\n10S\nHMFS0690-M20CF\nM20 x 1.5\n3/8”\n12S\nHMFS0690-M22CF\nM22 x 1.5\n3/8”\n14S\nHMFS0890-M24CF\nM24 x 1.5\n1/2”\n16S\nHMFS1090-M24CF\nM24 x 1.5\n5/8”\n16S\nHMFS1090-M30CF\nM30 x 2\n5/8”\n20S\nHMFS1290-M30CF\nM30 x 2\n3/4”\n20S\nHMFS1290-M36CF\nM36 x 2\n3/4”\n25S\nHMFS1690-M36CF\nM36 x 2\n1”\n25S\nHMFS1690-M42CF\nM42 x 2\n1”\n30S\nHMFS2090-M52CF\nM52 x 2\n1.1/4”\n38S\nHMFS2490-M52CF\nM52 x 2\n1.1/2”\n38S\nStandpipe, \nS Series\n0309\nOD mm\nTo Suit Hose ID\nHSSP04-10CF\n10\n1/4”\nHSSP06-14CF\n14\n3/8”\nHSSP08-16CF\n16\n1/2”\nHSSP10-20CF\n20\n5/8”\nHSSP12-20CF\n20\n3/4”\nHSSP12-25CF\n25\n3/4”\nHSSP16-25CF\n25\n1”\nHSSP16-30CF\n30\n1”\nHSSP20-38CF\n38\n1.1/4”\n45¡ Metric Female 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMFS03-45-M16CF\nM16 x 1.5\n3/16”\n8S\nHMFS04-45-M14CF\nM14 x 1.5\n1/4”\n6S\nHMFS04-45-M16CF\nM16 x 1.5\n1/4”\n8S\nHMFS04-45-M18CF\nM18 x 1.5\n1/4”\n10S\nHMFS05-45-M18CF\nM18 x 1.5\n5/16”\n10S\nHMFS05-45-M20CF\nM20 x 1.5\n5/16”\n12S\nHMFS06-45-M20CF\nM20 x 1.5\n3/8”\n12S\nHMFS06-45-M22CF\nM22 x 1.5\n3/8”\n14S\nHMFS08-45-M24CF\nM24 x 1.5\n1/2”\n16S\nHMFS10-45-M30CF\nM30 x 2\n5/8”\n20S\nHMFS12-45-M30CF\nM30 x 2\n3/4”\n20S\nHMFS12-45-M36CF\nM36 x 2\n3/4”\n25S\nHMFS16-45-M36CF\nM36 x 2\n1”\n25S\nHMFS16-45-M42CF\nM42 x 2\n1”\n30S\nHMFS20-45-M52CF\nM52 x 2\n1.1/4”\n38S\nHMFS24-45-M52CF\nM52 x 2\n1.1/2”\n38S\n90¡ Standpipe, \nS Series\n0309\nOD mm\nTo Suit Hose ID\nHSSP04-90-08CF\n8\n1/4”\nHSSP04-90-10CF\n10\n1/4”\nHSSP06-90-14CF\n14\n3/8”\nHSSP08-90-16CF\n16\n1/2”\nHSSP10-90-20CF\n20\n5/8”\nHSSP12-90-20CF\n20\n3/4”\nHSSP12-90-25CF\n25\n3/4”\nHSSP16-90-25CF\n25\n1”\nHSSP16-90-30CF\n30\n1”\n45¡ Standpipe, S Series\n0309\nOD mm\nTo Suit Hose ID\nHSSP04-45-08CF\n8\n1/4”\nHSSP04-45-10CF\n10\n1/4”\nHSSP04-45-12CF\n12\n1/4”\nHSSP05-45-12CF\n12\n5/16”\nHSSP06-45-14CF\n14\n3/8”\nHSSP08-45-14CF\n14\n1/2”\nHSSP08-45-16CF\n16\n1/2”\nHSSP10-45-16CF\n16\n5/8”\nHSSP10-45-20CF\n20\n5/8”\nHSSP12-45-20CF\n20\n3/4”\nHSSP12-45-25CF\n25\n3/4”\nHSSP16-45-25CF\n25\n1”\nHSSP16-45-30CF\n30\n1”\nHSSP16-45-38CF\n38\n1”\nHSSP20-45-30CF\n30\n1.1/4”\nHSSP20-45-38CF\n38\n1.1/4”\nHYDRAULIC HOSE CONNECTORS'
        ``````````````````````
      
        ``````````````````````                
        Ouput: ["Metric Male 24 Cone Seat,S Series" , "90 Metric Female 24􀂡 Cone Seat" , "S Series, DIN 3865" , "Metric Female 24 Cone Seat,S Series, DIN 3865" , "45 Metric Female 24 Cone Seat,S Series" , "45 Standpipe, S Series" , "90 Standpipe, S Series"]  
        ``````````````````````  
       
           """},
          {"role": "user", "content": f"""

        Now provide the output for
        

        ``````````````````````  
        Description of layout of table, header, sections and patterns:: {pattern_data}
        ``````````````````````

        ``````````````````````      
        Text document: {text_data}
        ``````````````````````
          """},
      ],
      response_format = response_format



)
  return completion.choices[0].message.parsed.output


# 2. LLM Parser
def llm_parser(user_text, target_table, target_variables, pattern, text_document, openai_client, response_format,  model = "gpt-4o"):
  completion = openai_client.beta.chat.completions.parse(
      model= model,
      messages=[
          {"role": "system", "content": f""" 
           
          You are a skilled data extraction and parsing specialist with expertise in analysing text documents and extracting structured information from tables. 
          
          Your job is to extract data form a target table based on the provided variable(s) from the said table in line with the user's request and return them in a format where each value is associated with each other.  

          Additionally i will provide input from another expert who has analysed the image and text of the document and will provide structural patterns and rules to help you parse effectively. 
        
          Example:
              
          ``````````````````````````
          User's request: " I would like to extract the Order No and Model from the pdf document. The order number looks something like this (1051042839) and the model No is an mixture of numbers and letters and looks like this for instance NLOLC1
          ``````````````````````````
           
 
          ``````````````````````````
          Structural patterns and rules: 'The provided text from the PDF page contains two main sections with tables:\n\n### Section 1: Steel Enclosed Contactors\n- **Structure and Headers:**\n  - Headers: "Order No.", "Model No.", "AC1 Heating & General Load", "Fluorescent", "Mercury", "Halogen", "LED", "No. of Poles", "Dimensions HxWxD"\n  - Each row lists specifications for steel-enclosed contactors, such as order and model numbers, electrical ratings (AC1, Fluorescent, etc.), the number of poles, and dimensions.\n\n### Section 2: Direct On Line, Reversing & Star Delta Starters\n- **Structure and Headers:**\n  - Headers: "Order No.", "Model No.", "Max Rating", "Control Voltage", "Dimensions H x W x D", "Type"\n  - Each row provides details on different types of starters, including order and model numbers, maximum rating, control voltage, dimensions, and type.\n\n### Parsing Pattern:\n- **Patterns to Extract:**\n  - Identify sections by headers that are distinct and capitalized or emphasized (e.g., "STEEL ENCLOSED CONTACTORS").\n  - Each header is followed by corresponding data in a structured row format.\n  - New entries often follow a pattern of numbers followed by letters (e.g., order numbers).\n  - Delimiters between rows are spaces or newline characters.\n\n### Grouping:\n- Each section\'s data is grouped under its title and is separated by bullet points or non-table descriptive text.\n- Recognize transitions between sections by change in headers or the start of bullet points.\n\nBased on the image, sections are distinct and outlined clearly. The headers are bold and in larger font, followed by tabulated, organized data. These visual cues correspond with text organization, ensuring accurate text extraction and data parsing.'

          ``````````````````````````  
                   
          ``````````````````````````
          Target table: STEEL ENCLOSED CONTACTORS
          ``````````````````````````
           
          ``````````````````````````
          Target variable(s): [model_number, order_number, heading]
          ``````````````````````````
        
          ``````````````````````````
          Text document: ['\n\n--- Page 3 ---\n\n\tSTEEL ENCLOSED CONTACTORS\nOrder No.\nModel No.\nAC1 Heating & \nGeneral Load\nFluorescent \nMercury\nHalogen\nLED\nNo. of Poles\nDimensions HxWxD\nSTEEL ENCLOSED CONTACTORS\n1051042857\nNLCONM25/4\n25A\n16A\n13A\n25A\n4 + N\n184 x 184 x 149mm\n1051042858\nNLCONM45/4\n40A\n27A\n22A\n40A\n4 + N\n184 x 184 x 149mm\n1051042859\nNLCONM63/3N\n60A\n40A\n32A\n60A\n3 + N\n305 x 285 x 159mm\n•\tManufactured in accordance with EN60947-4-1 \n•\tIP55 steel enclosure\n•\tStandard coil voltage: 230V 50/60Hz\n•\tAuxiliary contacts can be fitted\n•\tColour RAL7035 light grey\n•\tKnockouts top and bottom\n•\tAll units are suitable for two wire control i.e. time \nswitch, thermostat, sensor or other remote controls\n•\tSupplied complete with neutral terminals\nNLCONM100/3N\n\tDIRECT ON LINE, REVERSING & \nSTAR DELTA STARTERS\n•\tManufactured in accordance with EN60947-4-1 \n•\tIP55 steel enclosure\n•\tColour RAL7035 light grey\n•\tSurface mounting\n•\tStandard coil voltages: 230V or 400V 50/60Hz\n•\tBuilt in start/stop and reset push button\n•\tHand reset thermal overloads\n•\tOverload to be ordered separately\n•\tStainless steel pozidrive fixing screws for lid\n•\tKnockouts top and bottom\n•\tInternal components DIN rail mounted\nOrder No.\nModel No.\nMax Rating\nControl Voltage\nDimensions H x W x D\nType\nDIRECT ON LINE & REVERSING STARTERS\n1051042849\nNLSTM27C\n7.5kW\n230V\n174 x 104 x 134mm\nDirect on Line\n£0.00\n1051042850\nNLSTM47C\n7.5kW\n400V\n174 x 104 x 134mm\nDirect on Line\n£0.00\n1051042856\nNLRVM45C\n5.5kW\n400V\n184 x 184 x 149mm\nReversing\n£0.00\n1051042851\nNLSTM25CSW\n5.5kW\n230V\n184 x 184 x 149mm\nDOL With Isolator\n£0.00\n1051042852\nNLSTM45CSW\n5.5kW\n400V\n184 x 184 x 149mm\nDOL With Isolator\n£0.00\n1051042854\nNLSDM415C\n15kW\n400V\n184 x 284 x 149mm\nStar Delta\n£0.00\nNLSTM27C\nStock items delivered FREE anywhere in UK\n15 3\n\n',
          '\n']
          ``````````````````````````
           

          
          Output: 
          ``````````````````````````
          heading: STEEL ENCLOSED CONTACTORS, STEEL ENCLOSED CONTACTORS, STEEL ENCLOSED CONTACTORS
           
          order_number: 1051042857, 1051042858, 1051042859

          model_number: NLCONM25/4, NLCONM45/4, NLCONM63/3N

          ``````````````````````````

           
          NOTE: It is very important all variables have the same length of values. So where necessary repeat values for the required corresponding vlaues.
           """},
          {"role": "user", "content": f"""


          Now this is the user's request: 
           
          ``````````````````````````
          User's request: {user_text}
          ``````````````````````````

          ``````````````````````````
          Structural patterns and rules: {pattern}
          ``````````````````````````

          ``````````````````````````
          Target table: {target_table}
          ``````````````````````````

          ``````````````````````````
          Target variabe(s): {target_variables}
          ``````````````````````````

          ``````````````````````````
          Text document: {text_document}
          ``````````````````````````          

        
          """},
      ],
      response_format = response_format


)
  return completion


async def table_identification_llm(text_input, user_text, base64_image, open_api_key, model='gpt-4o'):
    try:
        # Prepare request headers and payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {open_api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
          You are an expert in PDF document analysis, table detection, and textual pattern recognition. 
          Your expertise lies in extracting structured information from PDF and identifying tables

          You will be provided with:
          * A user's request
          * Extracted text from a PDF page.
          * An image of the same page for reference.
 
          Your task is to:
          1. Determine and return the total number of tables present on the page in line with the user's request. 

          2. Identify and return tables present on the page in line with the user's request. You can use the table header for each table for this. If a table lacks explicit headers, provide a descriptive label for the table based on near by or other content that distinguishes it. Additionally kindly add it the position of the table on the page . Note: Seperate each table header or descriptor with " || " as a delimiter.
          
          Expected Output Format:

          1. Number of Tables on the Page: [Total count]  

          2. Table Headers:  [Table headers or table descriptors with their positions on the page delimited by " || " (This has to be one per table). eg: "Screw Machine -1 - Can be found on the bottom left position of the page" || "Screw Machine -2 - Can be found on the bottom right position of the page" || .....]  

          Note: If there are no tables on the page simply return "0" for the  Number of Tables on the Page and "no tables" for the Table Headers 
          Note: If the user requests to extract from a specific table, ensure you only return the tables that match the user's request. 
          
        ``````````````````````          
          User's request:    {user_text}
        ``````````````````````

        ``````````````````````
          Extracted text:   {text_input}
        ``````````````````````



                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        }

        # Use aiohttp to make the async request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as resp:
                response_json = await resp.json()

        # Parse response
        image_desc = response_json['choices'][0]['message']['content']
        return image_desc

    except Exception as e:
        print(f"Error occurred during API call: {e}")
        return 'error'





async def vision_column_llm_parser(user_text, text_input, table_to_target, base64_image, open_api_key, model='gpt-4o'):


        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {open_api_key}"
        }


        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""


          You are a skilled data extraction and parsing specialist with expertise in analysing PDF text and image documents and extracting structured information from tables.

          You will be provided with:

          * A users request. 
          * Table headers or descriptors with thier associated index
          * An image of the PDF page for reference.
          * All text data on the PDF page. 

       
        Your job is to read a user's request, the image of the page and the tex document. 
        You are to identify and extract the column names of each table specified in line with the user's request. 
        If there are multiple tables they will be sepreated with " || " .
        Column names refer to the labels assigned to each column with populated values. Columns names may be text, numerical, alphanumerial and could sometimes contian special characters.
        The user request may sometimes imply the extraction of select column names or generating new ones. e.g columns to flag if say prices are lower or higher than a certain value.
        Return these column names in a list with their index, corresponding table header or descriptors and example values per column and its location. 
        The location will let the user know if the table is exlusively located in the image or if it is also present in the text document and the image.("Table is only present in the image" or "Table is present in both the image and the text document")
        For a giving column name ensure the spaces within it are replaced with an under score for example (if the variable is "pack quantity" ensure you return "pack_qunatity") and ensure it is lower cased

        Example
        
        ``````````````````````
        User's request: I want to extract all data from the tables
        ``````````````````````
        
        
        ``````````````````````
        Table to extract the column names from: "[index 0: Metric Male 24 Cone Seat,S Series, DIN 3865 || index 1: Standpipe,S Series]"
        ``````````````````````
        
        ``````````````````````
        Text document: '\n\n--- Page 1 ---\n\n21\n1.01  Hydraulic Hose & Inserts\nHydraulics\n1\n Metric Heavy Series\n Hexavalent chromium free plating\nMetric Male 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMMS03-M16CF\nM16 x 1.5\n3/16”\n8S\nHMMS04-M14CF\nM14 x 1.5\n1/4”\n6S\nHMMS04-M16CF\nM16 x 1.5\n1/4”\n8S\nHMMS04-M18CF\nM18 x 1.5\n1/4”\n10S\nHMMS05-M18CF\nM18 x 1.5\n5/16”\n10S\nHMMS05-M20CF\nM20 x 1.5\n5/16”\n12S\nHMMS06-M18CF\nM18 x 1.5\n3/8”\n10S\nHMMS06-M20CF\nM20 x 1.5\n3/8”\n12S\nHMMS06-M22CF\nM22 x 1.5\n3/8”\n14S\nHMMS06-M24CF\nM24 x 1.5\n3/8”\n16S\nHMMS08-M24CF\nM24 x 1.5\n1/2”\n16S\nHMMS10-M30CF\nM30 x 2\n5/8”\n20S\nHMMS12-M30CF\nM30 x 2\n3/4”\n20S\nHMMS12-M36CF\nM36 x 2\n3/4”\n25S\nHMMS16-M36CF\nM36 x 1.5\n1”\n25S\nHMMS16-M42CF\nM42 x 2\n1”\n30S\nHMMS20-M42CF\nM42 x 2\n1.1/4”\n30S\nHMMS20-M52CF\nM52 x 2\n1.1/4”\n38S\nHMMS24-M52CF\nM52 x 2\n1.1/2”\n38S\nMetric Female 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMFS03-M16CF\nM16 x 1.5\n3/16”\n8S\nHMFS04-M14CF\nM14 x 1.5\n1/4”\n6S\nHMFS04-M16CF\nM16 x 1.5\n1/4”\n8S\nHMFS04-M18CF\nM18 x 1.5\n1/4”\n10S\nHMFS05-M18CF\nM18 x 1.5\n5/16”\n10S\nHMFS05-M20CF\nM20 x 1.5\n5/16”\n12S\nHMFS06-M16CF\nM16 x 1.5\n3/8”\n8S\nHMFS06-M18CF\nM18 x 1.5\n3/8”\n10S\nHMFS06-M20CF\nM20 x 1.5\n3/8”\n12S\nHMFS06-M22CF\nM22 x 1.5\n3/8”\n14S\nHMFS08-M22CF\nM22 x 1.5\n1/2”\n14S\nHMFS08-M24CF\nM24 x 1.5\n1/2”\n16S\nHMFS08-M30CF\nM30 x 2\n1/2”\n20S\nHMFS10-M24CF\nM24 x 1.5\n5/8”\n16S\nHMFS10-M30CF\nM30 x 2\n5/8”\n20S\nHMFS12-M30CF\nM30 x 2\n3/4”\n20S\nHMFS12-M36CF\nM36 x 2\n3/4”\n25S\nHMFS16-M36CF\nM36 x 2\n1”\n25S\nHMFS16-M42CF\nM42 x 2\n1”\n30S\nHMFS20-M52CF\nM52 x 2\n1.1/4”\n38S\nHMFS24-M52CF\nM52 x 2\n1.1/2”\n38S\n90¡ Metric Female 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMFS0390-M16CF\nM16 x 1.5\n3/16”\n8S\nHMFS0490-M14CF\nM14 x 1.5\n1/4”\n6S\nHMFS0490-M16CF\nM16 x 1.5\n1/4”\n8S\nHMFS0490-M18CF\nM18 x 1.5\n1/4”\n10S\nHMFS0490-M20CF\nM20 x 1.5\n1/4”\n12S\nHMFS0590-M18CF\nM18 x 1.5\n5/16”\n10S\nHMFS0590-M20CF\nM20 x 1.5\n5/16”\n12S\nHMFS0690-M18CF\nM18 x 1.5\n3/8”\n10S\nHMFS0690-M20CF\nM20 x 1.5\n3/8”\n12S\nHMFS0690-M22CF\nM22 x 1.5\n3/8”\n14S\nHMFS0890-M24CF\nM24 x 1.5\n1/2”\n16S\nHMFS1090-M24CF\nM24 x 1.5\n5/8”\n16S\nHMFS1090-M30CF\nM30 x 2\n5/8”\n20S\nHMFS1290-M30CF\nM30 x 2\n3/4”\n20S\nHMFS1290-M36CF\nM36 x 2\n3/4”\n25S\nHMFS1690-M36CF\nM36 x 2\n1”\n25S\nHMFS1690-M42CF\nM42 x 2\n1”\n30S\nHMFS2090-M52CF\nM52 x 2\n1.1/4”\n38S\nHMFS2490-M52CF\nM52 x 2\n1.1/2”\n38S\nStandpipe, \nS Series\n0309\nOD mm\nTo Suit Hose ID\nHSSP04-10CF\n10\n1/4”\nHSSP06-14CF\n14\n3/8”\nHSSP08-16CF\n16\n1/2”\nHSSP10-20CF\n20\n5/8”\nHSSP12-20CF\n20\n3/4”\nHSSP12-25CF\n25\n3/4”\nHSSP16-25CF\n25\n1”\nHSSP16-30CF\n30\n1”\nHSSP20-38CF\n38\n1.1/4”\n45¡ Metric Female 24¡ Cone Seat, \nS Series, DIN 3865\n0309\nThread\nTo Suit\nHose ID\nSize\nHMFS03-45-M16CF\nM16 x 1.5\n3/16”\n8S\nHMFS04-45-M14CF\nM14 x 1.5\n1/4”\n6S\nHMFS04-45-M16CF\nM16 x 1.5\n1/4”\n8S\nHMFS04-45-M18CF\nM18 x 1.5\n1/4”\n10S\nHMFS05-45-M18CF\nM18 x 1.5\n5/16”\n10S\nHMFS05-45-M20CF\nM20 x 1.5\n5/16”\n12S\nHMFS06-45-M20CF\nM20 x 1.5\n3/8”\n12S\nHMFS06-45-M22CF\nM22 x 1.5\n3/8”\n14S\nHMFS08-45-M24CF\nM24 x 1.5\n1/2”\n16S\nHMFS10-45-M30CF\nM30 x 2\n5/8”\n20S\nHMFS12-45-M30CF\nM30 x 2\n3/4”\n20S\nHMFS12-45-M36CF\nM36 x 2\n3/4”\n25S\nHMFS16-45-M36CF\nM36 x 2\n1”\n25S\nHMFS16-45-M42CF\nM42 x 2\n1”\n30S\nHMFS20-45-M52CF\nM52 x 2\n1.1/4”\n38S\nHMFS24-45-M52CF\nM52 x 2\n1.1/2”\n38S\n90¡ Standpipe, \nS Series\n0309\nOD mm\nTo Suit Hose ID\nHSSP04-90-08CF\n8\n1/4”\nHSSP04-90-10CF\n10\n1/4”\nHSSP06-90-14CF\n14\n3/8”\nHSSP08-90-16CF\n16\n1/2”\nHSSP10-90-20CF\n20\n5/8”\nHSSP12-90-20CF\n20\n3/4”\nHSSP12-90-25CF\n25\n3/4”\nHSSP16-90-25CF\n25\n1”\nHSSP16-90-30CF\n30\n1”\n45¡ Standpipe, S Series\n0309\nOD mm\nTo Suit Hose ID\nHSSP04-45-08CF\n8\n1/4”\nHSSP04-45-10CF\n10\n1/4”\nHSSP04-45-12CF\n12\n1/4”\nHSSP05-45-12CF\n12\n5/16”\nHSSP06-45-14CF\n14\n3/8”\nHSSP08-45-14CF\n14\n1/2”\nHSSP08-45-16CF\n16\n1/2”\nHSSP10-45-16CF\n16\n5/8”\nHSSP10-45-20CF\n20\n5/8”\nHSSP12-45-20CF\n20\n3/4”\nHSSP12-45-25CF\n25\n3/4”\nHSSP16-45-25CF\n25\n1”\nHSSP16-45-30CF\n30\n1”\nHSSP16-45-38CF\n38\n1”\nHSSP20-45-30CF\n30\n1.1/4”\nHSSP20-45-38CF\n38\n1.1/4”\nHYDRAULIC HOSE CONNECTORS'
        ``````````````````````
      
        Output:
        ``````````````````````                
        index: [0], table_header or descriptor: [Metric Male 24 Cone Seat,S Series, DIN 386], column_names: [0309, thread, to_suit_house_id, size], example_value_per_column: [0309: (HMMS03-M16CF) , thread: (M16 x 1.5), to_suit_house_id: (3/16”), size: (8S)], table_location: [Table is present in both the image and the text document]
        index: [1], table_header or descriptor: [Standpipe,S Series], column_names: [0309, od_mm, to_suit_house_id], example_value_per_column: [0309: (HSSP04-10CF), od_mm: (10), to_suit_house_id: (1/4”)], table_location: [Table is present in both the image and the text document]    
        ``````````````````````  
                  
        NOTE: Some columns have sub columns under them. Simply append the sub columns to the main column and return them as a unit. 
        NOTE: Ensure the table location is returned in line with the example format.
        Now provide output in line with the example format for the following: 
            
          ``````````````````````````
          User's request: {user_text}
          ``````````````````````````

          ``````````````````````````
          Table to extract data from: {table_to_target}
          ``````````````````````````

          ``````````````````````````
          Text document: {text_input}
          ``````````````````````````

                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}", "detail": "high"
                            }
                        }
                    ]
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload) as resp:
                 response_json = await resp.json()
                 image_desc = response_json['choices'][0]['message']['content']
                 return  image_desc



dict_example = """[
  {
    "0303": "100R1AT-20-40",
    "ID": "1.1/4\\"",
    "OD mm": "44.8",
    "Max Pressure Bar": "63",
    "Bend Radius mm": "420",
    "Burst Pressure Bar": "250"
  },
  {
    "0303": "100R1AT-24-40",
    "ID": "1.1/2\\"",
    "OD mm": "52",
    "Max Pressure Bar": "50",
    "Bend Radius mm": "500",
    "Burst Pressure Bar": "200"
  },
  {
    "0303": "100R1AT-32-40",
    "ID": "2\\"",
    "OD mm": "65.9",
    "Max Pressure Bar": "40",
    "Bend Radius mm": "630",
    "Burst Pressure Bar": "160"
  }
]"""


async def vision_llm_parser(user_text, text_input, table_to_target, base64_image, open_api_key, model='gpt-4o'):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {open_api_key}"
        }


        payload = {
            "model": model,
            # "temperature": temperature, 
            # "frequency_penalty": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""


          You are a skilled data extraction and parsing specialist with expertise in analysing PDF text and image documents and extracting structured information from tables into a format that can be be converted to a dataframe and then exported to excel.

          You will be provided with:

          * A users request. 
          * A table header or descriptor along with its position on the page to guide you on which table to extract data from.
          * An image of the PDF page for reference.
          * All text data on the PDF page. 
        
          I need to extract the data and return it in a dictionary format which can be converted to a dataframe and then exported to excel.

          Example:
          
          Input:
          ``````````````````````````
          User's request: " I would like to extract all data from the table.
          ``````````````````````````

          ``````````````````````````
          Table to extract data from: [1 Wire, 40 Metre Coils -  This can be found on the middle right position of the page]
          ``````````````````````````

          ``````````````````````````
          Text document: '\n\n--- Page 1 ---\n\n6\n1.01    Hydraulic Hose & Inserts\nHydraulics\n1\n EN853-1, DIN20022 (Exceeds SAE 100R1AT)\n1 Wire, 40 Metre Coils\n0303\nID\nOD mm\nMax\nPressure\nBar\nBend\nRadius\nmm\nBurst\nPressure\nBar\n100R1AT-20-40\n1.1/4”\n44.8\n63\n420\n250\n100R1AT-24-40\n1.1/2”\n52\n50\n500\n200\n100R1AT-32-40\n2”\n65.9\n40\n630\n160\nTech Sheet 5001\n1 Wire, 100 Metre Coils\n0303\nID\nOD mm\nMax\nPressure\nBar\nBend\nRadius\nmm\nBurst\nPressure\nBar\n100R1AT-03-100\n3/16”\n12.5\n250\n90\n1000\n100R1AT-04-100\n1/4”\n14.1\n225\n100\n900\n100R1AT-05-100\n5/16”\n15.7\n215\n115\n850\n100R1AT-06-100\n3/8”\n18.1\n180\n130\n720\n100R1AT-08-100\n1/2”\n21.4\n160\n180\n640\n100R1AT-10-100\n5/8”\n24.5\n130\n200\n520\n100R1AT-12-100\n3/4”\n28.5\n105\n240\n420\n100R1AT-16-100\n1”\n36.6\n88\n300\n350\nTech Sheet 5001\nApplication:\nMedium pressure hydraulic applications\nSpecim cation:\nDIN 20022, EN853 – 1 Exceeds 100 R1AT\nConstruction:\n Inner tube: oil resistant synthetic rubber (NBR)\nReinforcement: 1 single steel wire braid\nOuter cover: oil and weather resistant\nsynthetic rubber\nRecommended Fluids:\n Hose suitable for petroleum based\nhydraulic oil and water-glycol based\n. uids, air and water\nWorking Temperature:\n-40°C to +100°C\nNB:  For air above 17 bar,\nthe hose cover must\nbe pin-pricked\nAvailable per 1 metre\nSmooth cover available\non request\nHYDRAULIC HOSE RANGE\nPart\nNumber\nSize\nWorking \nPressure\nBurst \nPressure\nBend Radius\nSwaging Dimensions\n Tail ID Top \nTolerance\nTail ID \nBottom\nTolerance\nSwaged \nNo Go\nSwaged\nGo\nmm\ninch\nbar\npsi\nbar\npsi\nInch\nmm\nFT3 Type \nFerrule\nSwage Dia. \nin mm\nmm\nmm\nmm\nmm\n100R1AT-03\n4.7\n3/16\n250\n3625\n1000\n14503\n3.54\n90\nFT3-12-03CF\n15\n2.89\n3.05\n2.80\n2.93\n100R1AT-04\n6.3\n1/4\n225\n3260\n900\n13050\n3.94\n100\nFT3-12-04CF\n17.8\n3.86\n3.67\n3.67\n3.45\n100R1AT-05\n8\n5/16\n215\n3120\n850\n12325\n4.53\n115\nFT3-12-05CF\n20.5\n5.29\n5.11\n5.11\n4.91\n100R1AT-06\n10\n3/8\n180\n2610\n720\n10440\n5.12\n130\nFT3-12-06CF\n21.6\n6.78\n6.42\n6.42\n6.13\n100R1AT-08\n12.5\n1/2\n160\n2320\n640\n9280\n7.09\n180\nFT3-12-08CF\n24\n9.21\n8.86\n8.86\n8.4\n100R1AT-10\n16\n5/8\n130\n1885\n520\n7540\n7.87\n200\nFT3-12-10CF\n28.7\n12.09\n11.59\n11.59\n10.99\n100R1AT-12\n19\n3/4\n105\n1520\n420\n6090\n9.45\n240\nFT3-12-12CF\n31.3\n14.46\n13.91\n13.91\n13.31\n100R1AT-16\n25\n1\n88\n1280\n350\n5075\n11.81\n300\nFT3-12-16CF\n41\n18.89\n18.29\n18.29\n17.68\n100R1AT-20\n31.5\n1.1/4\n63\n910\n250\n3625\n16.54\n420\nFT3-12-20CF\n48.5\n25.4\n24.69\n24.69\n23.8\n100R1AT-24\n38\n1.1/2\n50\n725\n200\n2900\n19.69\n500\nFT3-12-24CF\n58.5\n31.4\n30.6\n30.6\n29.8\n100R1AT-32\n51\n2\n40\n580\n160\n2320\n24.8\n630\nFT3-12-32CF\n70\n41.4\n40.6\n40.6\n30.8\nHydraulic Hose, EN853, DIN20022, SAE100R1AT\nTechnical Data\n\n|-|+++|-|\n'
          ``````````````````````````

          Dictionary Output: 
          ``````````````````````````
        
          {dict_example}

          ``````````````````````````
          NOTE: Column names may be numerical (eg 0314, 4532, etc), textual or alphanumerical, ensure they are returned exactly as is.
          NOTE: Ensure the table extracted is as like for like as the table represented in the image.
          NOTE: Some tables may not have any clean column names. In such cases use your expertise to create suitable one. 
        
          Thinking step by step provide the output for the following:
          ``````````````````````````
          User's request: {user_text}
          ``````````````````````````
          
          ``````````````````````````
          Table to extract data from: {table_to_target}
          ``````````````````````````

          ``````````````````````````
          Text document: {text_input}
          ``````````````````````````

                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}", "detail": "high" # jpeg
                            }
                        }
                    ]
                }
            ]

        } 

        async with aiohttp.ClientSession() as session:
            async with session.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload) as resp:
                 response_json = await resp.json()
                 image_desc = response_json['choices'][0]['message']['content']
                 return  image_desc



