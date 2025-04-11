import pandas as pd
import os
import time
import json
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import the OpenAI module
try:
    from openai_ai import get_product_details_openai, get_matching_products_openai
except ImportError as e:
    print(f"ERROR: Could not import OpenAI module: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred when importing OpenAI module: {e}")
    traceback.print_exc()
    sys.exit(1)

# Create test directory if it doesn't exist
os.makedirs('mini_test_results', exist_ok=True)

# Create sample data
sample_ocr_text = [
    "Vasofix Safety FEP 14 G x 2\" (2.2 x 50 mm) IV Catheter Exp: 2026-07",
    "STERILE GLOVES Size 7.5 Powder Free 2026-05",
]

# Create or load OCR test data
test_ocr_file = 'mini_test_results/mini_ocr_text.csv'
if not os.path.exists(test_ocr_file):
    print(f"Creating sample OCR data file: {test_ocr_file}")
    ocr_df = pd.DataFrame({'ocr_text': sample_ocr_text, 'actual_index': [0, 1]})
    ocr_df.to_csv(test_ocr_file, index=False)
else:
    print(f"Loading existing OCR data from: {test_ocr_file}")
    ocr_csv = pd.read_csv(test_ocr_file)
    ocr_df = pd.DataFrame(ocr_csv)

# Create or load product database
test_products_file = 'mini_test_results/mini_products.csv'
if not os.path.exists(test_products_file):
    print(f"Creating sample products database: {test_products_file}")
    products_data = [
        {'name': 'Vasofix Safety IV Catheter 14G', 'category': 'A', 'expiration_date': '2026-07-15', 'quantity': 1},
        {'name': 'Vasofix Safety IV Catheter 16G', 'category': 'A', 'expiration_date': '2026-04-20', 'quantity': 1},
        {'name': 'Sterile Gloves Size 7.5', 'category': 'B', 'expiration_date': '2026-05-10', 'quantity': 1},
        {'name': 'Sterile Gloves Size 8.0', 'category': 'B', 'expiration_date': '2026-03-11', 'quantity': 1},
    ]
    existing_products_df = pd.DataFrame(products_data)
    existing_products_df.to_csv(test_products_file, index=True)
else:
    print(f"Loading existing products database from: {test_products_file}")
    existing_products_csv = pd.read_csv(test_products_file)
    existing_products_df = pd.DataFrame(existing_products_csv)

# Initialize result DataFrame
results_df = ocr_df.copy()
results_df['matched_correctly'] = 0
results_df['extracted_name'] = ''
results_df['extracted_expiration'] = ''
results_df['matched_indices'] = ''

def convert_json_to_pd(json_data):
    try:
        data_dict = json.loads(json_data)
        df = pd.DataFrame(data_dict['products'])
        return df
    except Exception as e:
        print(f"ERROR: Failed to convert JSON to DataFrame: {e}")
        print(f"JSON data: {json_data}")
        raise

def convert_json_to_dict(json_data):
    try:
        data_dict = json.loads(json_data)
        return data_dict
    except Exception as e:
        print(f"ERROR: Failed to convert JSON to dictionary: {e}")
        print(f"JSON data: {json_data}")
        raise

def test_openai():
    print("\n" + "="*80)
    print("STARTING OPENAI TEST")
    print("="*80)
    
    top_n = 3
    total_first_time = 0
    total_second_time = 0
    success_count = 0
    count_correct = 0
    
    for i, row in ocr_df.iterrows():
        ocr_text = row['ocr_text']
        actual_index = row['actual_index']
        print(f"\nProcessing OCR text [{i+1}/{len(ocr_df)}]: {ocr_text}")
        
        try:
            # First API call - Get product details
            print("Making first API call to extract product details...")
            start_time = time.time()
            product_details_json = get_product_details_openai(ocr_text)
            end_time = time.time()
            first_time = end_time - start_time
            total_first_time += first_time
            
            product_details_df = convert_json_to_pd(product_details_json)
            
            product_name = product_details_df['name'].values[0]
            expiration_date = product_details_df['expiration_date'].values[0]
            
            # Store extracted info in results
            results_df.at[i, 'extracted_name'] = product_name
            results_df.at[i, 'extracted_expiration'] = expiration_date
            
            print(f"Extracted product: {product_name}")
            print(f"Extracted expiration date: {expiration_date}")
            
            # Second API call - Match with existing products
            print("\nMaking second API call to match with existing products...")
            start_time = time.time()
            matching_products_json = get_matching_products_openai(product_name, expiration_date, top_n, existing_products_df)
            end_time = time.time()
            second_time = end_time - start_time
            total_second_time += second_time
            
            matched_dict = convert_json_to_dict(matching_products_json)
            matched_indices = [p.get('index') for p in matched_dict.get('products', [])]
            
            # Store matched indices as a string
            results_df.at[i, 'matched_indices'] = ', '.join(map(str, matched_indices))
            
            # Check if actual index is in the matches
            if actual_index in matched_indices:
                results_df.at[i, 'matched_correctly'] = 1
                count_correct += 1
            
            print(f"Matched {len(matched_dict.get('products', []))} products with indices: {matched_indices}")
            for j, product in enumerate(matched_dict.get('products', [])):
                print(f"  {j+1}. Index: {product.get('index')}, Name: {product.get('name')}")
            
            print(f"First API call took {first_time:.2f}s, Second API call took {second_time:.2f}s")
            success_count += 1
            
        except Exception as e:
            print(f"ERROR processing OCR text: {e}")
            traceback.print_exc()
        
        print("-" * 80)
    
    # Calculate accuracy
    accuracy = count_correct / len(ocr_df) if len(ocr_df) > 0 else 0
    
    print("\nRESULTS SUMMARY:")
    print(f"Total samples: {len(ocr_df)}")
    print(f"Successfully processed: {success_count}/{len(ocr_df)}")
    print(f"Correctly matched: {count_correct}/{len(ocr_df)}")
    print(f"Total first API call time: {total_first_time:.2f}s")
    print(f"Total second API call time: {total_second_time:.2f}s")
    print(f"Total time: {total_first_time + total_second_time:.2f}s")
    print(f"Average time per sample: {(total_first_time + total_second_time)/len(ocr_df):.2f}s")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Save results
    results_file = 'mini_test_results/mini_results_simplified.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Detailed results saved to: {results_file}")
    
    summary_df = pd.DataFrame({
        'ai_model': ['openai'],
        'accuracy': [accuracy],
        'time_first_api_call': [total_first_time],
        'time_second_api_call': [total_second_time],
        'time_total': [total_first_time + total_second_time],
        'cost': ['']
    })
    
    summary_file = 'mini_test_results/mini_summary_simplified.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary results saved to: {summary_file}")
    
    return accuracy, total_first_time, total_second_time

if __name__ == "__main__":
    try:
        print(f"Started mini test at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        test_openai()
        print(f"Completed mini test at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1) 