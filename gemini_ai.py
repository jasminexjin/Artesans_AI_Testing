from typing import List
import httpx
import google.genai as genai
from google.genai import types
import json
import os
import dotenv
import pandas as pd
import time
from prompt import get_ocr_prompt, get_matching_products_prompt

dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')


#initialize the genai client
client =genai.Client(api_key= GEMINI_API_KEY)

def get_product_details_gemini(ocr_text: str):
    try:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        prompt = get_ocr_prompt(ocr_text)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if not response:
            raise ValueError("No response from Gemini API")
        
        if not hasattr(response, "text") or not response.text:
            raise ValueError("Empty response from Gemini API")
        
        # Parse the JSON response
        try:
            # The response is already a Python object, so no need to parse it
            response_data = response.text
            
            # Convert to a proper JSON string for compatibility with other code
            if isinstance(response_data, (dict, list)):
                # If Gemini returned a Python object directly, convert to JSON string
                if isinstance(response_data, list):
                    # Wrap the list in a products dictionary if needed
                    response_json = json.dumps({"products": response_data})
                else:
                    # If it's already a dict, just convert to string
                    response_json = json.dumps(response_data)
            else:
                # If it's already a string, make sure it's valid JSON
                try:
                    json.loads(response_data)  # Just to validate
                    response_json = response_data
                except json.JSONDecodeError:
                    # If not valid JSON, fallback to a default structure
                    response_json = json.dumps({"products": [{"name": "Unknown", "expiration_date": "Unknown"}]})
            
            return response_json
            
        except Exception as e:
            raise ValueError(f"Failed to process response: {str(e)}")
    
    except Exception as e:
        print(f"Error in get_product_details: {str(e)}")
        raise Exception(f"Failed to process OCR text: {str(e)}")
        

#Call Gemini AI API to find matching products based on product name and expiration date
def get_matching_products_gemini(product_name: str, expiration_date: str, top_n: int, existing_products: pd.DataFrame):
    
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not found in environment variables")
    
    prompt = get_matching_products_prompt(product_name, expiration_date, top_n, existing_products)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        if not response:
            raise Exception("No response from Gemini API")
        
  
        if not hasattr(response, "text") or not response.text:
            raise ValueError("Empty response from Gemini API")
        
        # Process the response
        try:
            # The response is already a Python object, so no need to parse it
            response_data = response.text
            
            # Convert to a proper JSON string for compatibility with other code
            if isinstance(response_data, (dict, list)):
                # If Gemini returned a Python object directly, convert to JSON string
                if isinstance(response_data, list):
                    # Wrap the list in a products dictionary if needed
                    response_json = json.dumps({"products": response_data})
                else:
                    # If it's already a dict, just convert to string
                    response_json = json.dumps(response_data)
            else:
                # If it's already a string, make sure it's valid JSON
                try:
                    json.loads(response_data)  # Just to validate
                    response_json = response_data
                except json.JSONDecodeError:
                    # If not valid JSON, fallback to a default structure
                    response_json = json.dumps({"products": []})
            
            return response_json
            
        except Exception as e:
            raise ValueError(f"Failed to process response: {str(e)}")
        
    except Exception as e:
        raise Exception(f"Gemini API call failed: {str(e)}") 
    
def test_gemini_ai():
    # Test OCR text extraction
    test_ocr_text = """
    Product Name: Test Product
    Expiration Date: 2024-12-31
    Quantity: 100
    """
    
    ocr_result = get_product_details_gemini(test_ocr_text)
    print(f"OCR Result: {ocr_result}")
    
    # Validate it's a string that can be parsed as JSON
    assert isinstance(ocr_result, str), "OCR result should be a JSON string"
    ocr_dict = json.loads(ocr_result)
    assert "products" in ocr_dict, "OCR result should have a 'products' key"
    
    # Test product matching
    test_products = pd.DataFrame({
        'name': ['Test Product 1', 'Test Product 2'],
        'expiration_date': ['2024-12-31', '2024-12-31']
    })
    
    matching_result = get_matching_products_gemini(
        product_name="Test Product",
        expiration_date="2024-12-31",
        top_n=1,
        existing_products=test_products
    )
    print(f"Matching Result: {matching_result}")
    
    # Validate it's a string that can be parsed as JSON
    assert isinstance(matching_result, str), "Matching result should be a JSON string"
    match_dict = json.loads(matching_result)
    assert "products" in match_dict, "Matching result should have a 'products' key"
    
    print("All Gemini AI tests passed!")


if __name__ == "__main__":
    test_gemini_ai()