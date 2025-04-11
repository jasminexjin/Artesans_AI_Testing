from openai import OpenAI
import os
import dotenv
import pandas as pd
from prompt import get_ocr_prompt, get_matching_products_prompt

dotenv.load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_product_details_openai(ocr_text):
    prompt = get_ocr_prompt(ocr_text)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content

def get_matching_products_openai(product_name, expiration_date, top_n, existing_products):
    prompt = get_matching_products_prompt(product_name, expiration_date, top_n, existing_products)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def test_openai_ai():
    # Test OCR text extraction
    test_ocr_text = """
    Product Name: Test Product
    Expiration Date: 2024-12-31
    Quantity: 100
    """
    
    ocr_result = get_product_details_openai(test_ocr_text)
    assert ocr_result is not None, "OCR extraction failed"
    
    # Test product matching
    test_products = pd.DataFrame({
        'name': ['Test Product 1', 'Test Product 2'],
        'expiration_date': ['2024-12-31', '2024-12-31']
    })
    
    matching_result = get_matching_products_openai(
        product_name="Test Product",
        expiration_date="2024-12-31",
        top_n=1,
        existing_products=test_products
    )
    assert matching_result is not None, "Product matching failed"
    
    print(f"All OpenAI AI tests passed!, {ocr_result}, {matching_result}")

#print(test_openai_ai())
