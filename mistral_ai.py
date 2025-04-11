import os
from dotenv import load_dotenv
from mistralai import Mistral
import pandas as pd
from prompt import get_ocr_prompt, get_matching_products_prompt
import json

load_dotenv()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
MISTRAL_MODEL = os.getenv('MISTRAL_MODEL')

client = Mistral(api_key=MISTRAL_API_KEY)

def get_product_details_mistral(ocr_text):
    prompt = get_ocr_prompt(ocr_text)
    prompt += "\nIMPORTANT: Return ONLY the JSON object with no explanation or markdown formatting. Do not use ```json or ``` markers."
    
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_matching_products_mistral(product_name, expiration_date, top_n, existing_products):
    prompt = get_matching_products_prompt(product_name, expiration_date, top_n, existing_products)
    prompt += "\nIMPORTANT: Return ONLY the JSON object with no explanation or markdown formatting. Do not use ```json or ``` markers."
    
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "user", "content": prompt} 
        ]
    )
    return response.choices[0].message.content

def test_mistral_ai():
    # Test OCR text extraction
    test_ocr_text = """
    Product Name: Test Product
    Expiration Date: 2024-12-31
    Quantity: 100
    """
    
    ocr_result = get_product_details_mistral(test_ocr_text)
    ocr_dict = json.loads(ocr_result)
    ocr_df = pd.DataFrame(ocr_dict)
    assert ocr_df is not None, "OCR extraction failed"
    
    # Test product matching
    test_products = pd.DataFrame({
        'name': ['Test Product 1', 'Test Product 2'],
        'expiration_date': ['2024-12-31', '2024-12-31']
    })
    
    matching_result = get_matching_products_mistral(
        product_name="Test Product",
        expiration_date="2024-12-31",
        top_n=1,
        existing_products=test_products
    )
    match_dict = json.loads(matching_result)
    assert match_dict is not None, "Product matching failed"
    
    print(f"All Mistral AI tests passed! {match_dict}")

'''
if __name__ == "__main__":
    test_mistral_ai()
'''