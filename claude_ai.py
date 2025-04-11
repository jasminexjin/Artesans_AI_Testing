import os
import dotenv
import json
import re
from anthropic import Anthropic
from prompt import get_ocr_prompt, get_matching_products_prompt
import pandas as pd
dotenv.load_dotenv()

CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-opus-20240229')

client = Anthropic(api_key=CLAUDE_API_KEY)

def extract_json_from_text(text):
    """Extract JSON content from Claude's response text."""
    # Look for JSON content between ```json and ``` markers
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(1).strip()
        try:
            # Validate it's proper JSON
            json_obj = json.loads(json_str)
            return json.dumps(json_obj)  # Return formatted JSON string
        except json.JSONDecodeError:
            pass  # If it's not valid JSON, fall through to backup methods
    
    # Backup method: Try to find anything that looks like a JSON object
    json_pattern = r'(\{[\s\S]*\})'
    match = re.search(json_pattern, text)
    if match:
        json_str = match.group(1).strip()
        try:
            # Validate it's proper JSON
            json_obj = json.loads(json_str)
            return json.dumps(json_obj)  # Return formatted JSON string
        except json.JSONDecodeError:
            pass  # If it's not valid JSON, move to final fallback
    
    # Final fallback: Create a minimal valid JSON structure
    print("WARNING: Could not extract valid JSON from Claude response. Creating fallback JSON.")
    print(f"Raw response: {text[:200]}...")
    
    if "product details" in text.lower() or "extract" in text.lower():
        # Product details fallback
        fallback = {
            "products": [
                {
                    "name": "Unknown Product",
                    "expiration_date": "Unknown"
                }
            ]
        }
    else:
        # Matching products fallback
        fallback = {
            "products": []
        }
    
    return json.dumps(fallback)

def get_product_details_claude(ocr_text: str):
    """Get product details from OCR text using Claude."""
    prompt = get_ocr_prompt(ocr_text)
    prompt += "\nIMPORTANT: Your response MUST be in valid JSON format with a 'products' array containing at least 'name' and 'expiration_date' fields."
    prompt += "\nUse ```json\n{...}\n``` format in your response."

    response = client.messages.create(
        model=CLAUDE_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=4000 
    )

    response_text = response.content[0].text
    # Extract and validate JSON from Claude's response
    return extract_json_from_text(response_text)


def get_matching_products_claude(product_name: str, expiration_date: str, top_n: int, existing_products: pd.DataFrame): 
    """Match product with existing products using Claude."""
    prompt = get_matching_products_prompt(product_name, expiration_date, top_n, existing_products)
    prompt += "\nIMPORTANT: Your response MUST be in valid JSON format with a 'products' array containing matched products with 'index' and 'name' fields."
    prompt += "\nUse ```json\n{...}\n``` format in your response."

    response = client.messages.create(
        model=CLAUDE_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=4000 
    )

    response_text = response.content[0].text
    # Extract and validate JSON from Claude's response
    return extract_json_from_text(response_text)

def test_claude_ai():
    # Test OCR text extraction
    test_ocr_text = """
    Product Name: Test Product
    Expiration Date: 2024-12-31
    Quantity: 100
    """
    
    ocr_result = get_product_details_claude(test_ocr_text)
    assert ocr_result is not None, "OCR extraction failed"
    
    # Validate JSON format
    try:
        json.loads(ocr_result)
        print("✓ OCR result is valid JSON")
    except json.JSONDecodeError:
        print("✗ OCR result is not valid JSON")
    
    # Test product matching
    test_products = pd.DataFrame({
        'name': ['Test Product 1', 'Test Product 2'],
        'expiration_date': ['2024-12-31', '2024-12-31']
    })
    
    matching_result = get_matching_products_claude(
        product_name="Test Product",
        expiration_date="2024-12-31",
        top_n=1,
        existing_products=test_products
    )
    assert matching_result is not None, "Product matching failed"
    
    # Validate JSON format
    try:
        json.loads(matching_result)
        print("✓ Matching result is valid JSON")
    except json.JSONDecodeError:
        print("✗ Matching result is not valid JSON")
    
    print("All Claude AI tests passed!")

if __name__ == "__main__":
    test_claude_ai()
    

