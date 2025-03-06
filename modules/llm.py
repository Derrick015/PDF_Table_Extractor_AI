from pydantic import BaseModel
import aiohttp


async def table_identification_llm(text_input, user_text, base64_image, open_api_key, model='gpt-4o'):
    """
    Calls the OpenAI API to identify tables in a PDF page using both text and image data.
    
    Parameters:
        text_input (str): Extracted text from the PDF page
        user_text (str): User's request or instructions
        base64_image (str): Base64-encoded image of the PDF page
        open_api_key (str): OpenAI API key
        model (str): OpenAI model to use, defaults to 'gpt-4o'
        
    Returns:
        str: The response text from the LLM containing table identification information
        
    Raises:
        Exception: If there's an error in the API call
    """
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
    """
    Calls the OpenAI API to extract and parse table data from a PDF page using both text and image data.
    
    Parameters:
        user_text (str): User's request or instructions
        text_input (str): Extracted text from the PDF page
        table_to_target (str): The specific table header to target for extraction
        base64_image (str): Base64-encoded image of the PDF page
        open_api_key (str): OpenAI API key
        model (str): OpenAI model to use, defaults to 'gpt-4o'
        
    Returns:
        str: The response text from the LLM containing parsed table data
        
    Raises:
        Exception: If there's an error in the API call
    """
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
          Text document: '\n\n--- Page 1 ---\n\n6\n1.01    Hydraulic Hose & Inserts\nHydraulics\n1\n EN853-1, DIN20022 (Exceeds SAE 100R1AT)\n1 Wire, 40 Metre Coils\n0303\nID\nOD mm\nMax\nPressure\nBar\nBend\nRadius\nmm\nBurst\nPressure\nBar\n100R1AT-20-40\n1.1/4"\n44.8\n63\n420\n250\n100R1AT-24-40\n1.1/2"\n52\n50\n500\n200\n100R1AT-32-40\n2"\n65.9\n40\n630\n160\nTech Sheet 5001\n1 Wire, 100 Metre Coils\n0303\nID\nOD mm\nMax\nPressure\nBar\nBend\nRadius\nmm\nBurst\nPressure\nBar\n100R1AT-03-100\n3/16"\n12.5\n250\n90\n1000\n100R1AT-04-100\n1/4"\n14.1\n225\n100\n900\n100R1AT-05-100\n5/16"\n15.7\n215\n115\n850\n100R1AT-06-100\n3/8"\n18.1\n180\n130\n720\n100R1AT-08-100\n1/2"\n21.4\n160\n180\n640\n100R1AT-10-100\n5/8"\n24.5\n130\n200\n520\n100R1AT-12-100\n3/4"\n28.5\n105\n240\n420\n100R1AT-16-100\n1"\n36.6\n88\n300\n350\nTech Sheet 5001\nApplication:\nMedium pressure hydraulic applications\nSpecim cation:\nDIN 20022, EN853 – 1 Exceeds 100 R1AT\nConstruction:\n Inner tube: oil resistant synthetic rubber (NBR)\nReinforcement: 1 single steel wire braid\nOuter cover: oil and weather resistant\nsynthetic rubber\nRecommended Fluids:\n Hose suitable for petroleum based\nhydraulic oil and water-glycol based\n. uids, air and water\nWorking Temperature:\n-40°C to +100°C\nNB:  For air above 17 bar,\nthe hose cover must\nbe pin-pricked\nAvailable per 1 metre\nSmooth cover available\non request\nHYDRAULIC HOSE RANGE\nPart\nNumber\nSize\nWorking \nPressure\nBurst \nPressure\nBend Radius\nSwaging Dimensions\n Tail ID Top \nTolerance\nTail ID \nBottom\nTolerance\nSwaged \nNo Go\nSwaged\nGo\nmm\ninch\nbar\npsi\nbar\npsi\nInch\nmm\nFT3 Type \nFerrule\nSwage Dia. \nin mm\nmm\nmm\nmm\nmm\n100R1AT-03\n4.7\n3/16\n250\n3625\n1000\n14503\n3.54\n90\nFT3-12-03CF\n15\n2.89\n3.05\n2.80\n2.93\n100R1AT-04\n6.3\n1/4\n225\n3260\n900\n13050\n3.94\n100\nFT3-12-04CF\n17.8\n3.86\n3.67\n3.67\n3.45\n100R1AT-05\n8\n5/16\n215\n3120\n850\n12325\n4.53\n115\nFT3-12-05CF\n20.5\n5.29\n5.11\n5.11\n4.91\n100R1AT-06\n10\n3/8\n180\n2610\n720\n10440\n5.12\n130\nFT3-12-06CF\n21.6\n6.78\n6.42\n6.42\n6.13\n100R1AT-08\n12.5\n1/2\n160\n2320\n640\n9280\n7.09\n180\nFT3-12-08CF\n24\n9.21\n8.86\n8.86\n8.4\n100R1AT-10\n16\n5/8\n130\n1885\n520\n7540\n7.87\n200\nFT3-12-10CF\n28.7\n12.09\n11.59\n11.59\n10.99\n100R1AT-12\n19\n3/4\n105\n1520\n420\n6090\n9.45\n240\nFT3-12-12CF\n31.3\n14.46\n13.91\n13.91\n13.31\n100R1AT-16\n25\n1\n88\n1280\n350\n5075\n11.81\n300\nFT3-12-16CF\n41\n18.89\n18.29\n18.29\n17.68\n100R1AT-20\n31.5\n1.1/4\n63\n910\n250\n3625\n16.54\n420\nFT3-12-20CF\n48.5\n25.4\n24.69\n24.69\n23.8\n100R1AT-24\n38\n1.1/2\n50\n725\n200\n2900\n19.69\n500\nFT3-12-24CF\n58.5\n31.4\n30.6\n30.6\n29.8\n100R1AT-32\n51\n2\n40\n580\n160\n2320\n24.8\n630\nFT3-12-32CF\n70\n41.4\n40.6\n40.6\n30.8\nHydraulic Hose, EN853, DIN20022, SAE100R1AT\nTechnical Data\n\n|-|+++|-|\n'
          ``````````````````````````

          Dictionary Output: 
          ``````````````````````````
        
          {dict_example}

          ``````````````````````````
          NOTE: Column names may be numerical (eg 0314, 4532, etc), textual or alphanumerical, ensure they are returned exactly as is unless specified otherwise by the user.
          NOTE: Ensure the table extracted is as like for like as the table represented in the image.
          NOTE: Some columns may have multiple sub columns under them. Simply append the sub columns to the main column and return them as the column name unless specified otherwise by the user.
        
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
                            "url": f"data:image/png;base64,{base64_image}", "detail": "auto" # jpeg
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