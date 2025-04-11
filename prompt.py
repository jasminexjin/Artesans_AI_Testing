import pandas as pd
def get_ocr_prompt(ocr_text: str):
    
    prompt =f"""
        Please return JSON describing the product name with product specification, expiration date, and quantity of the products using the following schema:

        

        PRODUCT = {{
            "name": str, 
            "category": str,
            "expiration_date": str, 
            "quantity": int,
        }}

        **Rules:**
        - "name": Extract the full product name and classify the item (e.g., "IV Catheter", "Syringe", "Gloves", "Bandage").
          Ensure it includes specifications (e.g., gauge, size, type). If a number is next to units like "ml", "G", or "mm", add a space before the unit.
          Translate any non-English text to **English**.
        - "category": Classify based on its **Category** in medical terms (e.g., A, B, C, D, E, Assorted).
        - "expiration_date": Locate the **hourglass icon** on packaging. Extract the date **in YYYY-MM or YYYY-MM-DD format**.
          If 6 digits, assume **YYYYMM**. Ignore years **before 2025**.
        - "quantity": An integer. If missing, return **1**.

        **Example Output:**
        ```json
        {{
            "products": [
                {{
                    "name": "Vasofix Safety FEP 14 G x 2\" (2.2 x 50 mm) - IV Catheter",
                    "category": "A",
                    "expiration_date": "2026-07-01",
                    "quantity": 1
                }}
            ]
        }}
        ```
        
        Here is the provided text extracted via OCR:
        {ocr_text}
        
        Return 'products': list[PRODUCT]
        
        """
    
    return prompt


def get_matching_products_prompt(product_name: str, expiration_date: str, top_n: int, existing_products: pd.DataFrame):

    product_list = "\n".join([
        f"index {index}: {row['name']} (expiration_date: {row['expiration_date'] if pd.notna(row['expiration_date']) else 'Unknown'})"
        for index, row in existing_products.iterrows()
    ])

    prompt = f"""
    You are a product-matching assistant.

    Given a product with:   
    - name: '{product_name}'
    - expiration_date: '{expiration_date}'


    Find the best matching product(s) from the list below:

    {product_list}

    Return the top {top_n} closest matches in **JSON format** as a list called 'products', where each item includes:
    - 'index': int (index from the existing_products list)
    - 'name': str
    - 'category': str
    - 'expiration_date': str
    - 'quantity': int
    - 'comments': str

    Example response format:
    {{
      "products": [
        {{
            "index": 0,
            "name": "IV Catheter 16G",
            "expiration_date": "2025-06-30",
            "quantity": 5,
            "comments": "This is a comment"
        }},
        ...
    ]
    }}

    Prioritize semantic similarity in product names. If the expiration date is provided, give higher weight to products with matching or nearby dates.
    Only return the 'products' list. No explanation, no text outside the JSON.
    """

    return prompt