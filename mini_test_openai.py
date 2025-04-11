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

# Load OCR test data from specified file
ocr_file = 'Testing/ocr_text.csv'
print(f"Loading OCR data from: {ocr_file}")
try:
    ocr_df = pd.read_csv(ocr_file)
    # Add actual_index column if it doesn't exist
    if 'actual_index' not in ocr_df.columns:
        print("Adding 'actual_index' column to OCR data")
        ocr_df['actual_index'] = range(len(ocr_df))
except FileNotFoundError:
    print(f"ERROR: OCR file not found: {ocr_file}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to load OCR data: {e}")
    traceback.print_exc()
    sys.exit(1)

# Load product database from Excel file
products_file = 'Testing/full_data_cleaned.xlsx'
print(f"Loading products database from: {products_file}")
try:
    existing_products_df = pd.read_excel(products_file)
except FileNotFoundError:
    print(f"ERROR: Products file not found: {products_file}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to load products data: {e}")
    traceback.print_exc()
    sys.exit(1)

# Initialize DataFrame to store matched products
matching_products_df = ocr_df.copy()
matching_products_df['matched_dict'] = None
matching_products_df['matched_dict'] = matching_products_df['matched_dict'].astype('object')

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

def get_accuracy(ocr_df, matching_products_df):
    try:
        count_correct = 0
        final_results_df = ocr_df.copy()
        final_results_df['if_matched_correctly'] = 0
        
        for i, row in ocr_df.iterrows():
            ocr_text = row['ocr_text']
            actual_index = row['actual_index']
            matched_dict = matching_products_df.loc[matching_products_df['ocr_text'] == ocr_text]
            
            if len(matched_dict) == 0 or matched_dict['matched_dict'].iloc[0] is None:
                print(f"WARNING: No match found for OCR text: {ocr_text}")
                continue
                
            matched_indices = [p.get('index') for p in matched_dict['matched_dict'].iloc[0].get('products', [])]
            
            if actual_index in matched_indices:
                final_results_df.loc[i, 'if_matched_correctly'] = 1
                count_correct += 1
                
        total_accuracy = count_correct / len(ocr_df) if len(ocr_df) > 0 else 0
        return total_accuracy, final_results_df
    except Exception as e:
        print(f"ERROR: Failed to calculate accuracy: {e}")
        traceback.print_exc()
        return 0, ocr_df.copy()

def test_openai():
    print("\n" + "="*80)
    print("STARTING OPENAI TEST")
    print("="*80)
    
    top_n = 3
    total_first_time = 0
    total_second_time = 0
    success_count = 0
    
    for i, row in ocr_df.iterrows():
        ocr_text = row['ocr_text']
        print(f"\nProcessing OCR text [{i+1}/{len(ocr_df)}]: {ocr_text}")
        
        try:
            # First API call - Get product details
            print("Making first API call to extract product details...")
            start_time = time.time()
            product_details_json = get_product_details_openai(ocr_text)
            end_time = time.time()
            first_time = end_time - start_time
            total_first_time += first_time
            
            print(f"Raw JSON response: {product_details_json}")
            product_details_df = convert_json_to_pd(product_details_json)
            
            product_name = product_details_df['name'].values[0]
            expiration_date = product_details_df['expiration_date'].values[0]
            
            print(f"Extracted product: {product_name}")
            print(f"Extracted expiration date: {expiration_date}")
            
            time.sleep(10)
            # Second API call - Match with existing products
            print("\nMaking second API call to match with existing products...")
            start_time = time.time()
            matching_products_json = get_matching_products_openai(product_name, expiration_date, top_n, existing_products_df)
            end_time = time.time()
            second_time = end_time - start_time
            total_second_time += second_time
            
            print(f"Raw JSON response: {matching_products_json}")
            matched_dict = convert_json_to_dict(matching_products_json)
            
            # Store the matched dictionary in the DataFrame
            matching_products_df.at[i, 'matched_dict'] = matched_dict
            
            print(f"Matched {len(matched_dict.get('products', []))} products")
            for j, product in enumerate(matched_dict.get('products', [])):
                print(f"  {j+1}. Index: {product.get('index')}, Name: {product.get('name')}")
            
            print(f"First API call took {first_time:.2f}s, Second API call took {second_time:.2f}s")
            success_count += 1
            time.sleep(30)
            
        except Exception as e:
            print(f"ERROR processing OCR text: {e}")
            traceback.print_exc()
            matching_products_df.at[i, 'matched_dict'] = {"products": []}
        
        print("-" * 80)
    
    # Calculate accuracy
    print("\nCalculating accuracy...")
    accuracy, results_df = get_accuracy(ocr_df, matching_products_df)
    
    print("\nRESULTS SUMMARY:")
    print(f"Total samples: {len(ocr_df)}")
    print(f"Successfully processed: {success_count}/{len(ocr_df)}")
    print(f"Total first API call time: {total_first_time:.2f}s")
    print(f"Total second API call time: {total_second_time:.2f}s")
    print(f"Total time: {total_first_time + total_second_time:.2f}s")
    print(f"Average time per sample: {(total_first_time + total_second_time)/len(ocr_df):.2f}s")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Save results
    results_file = 'mini_test_results/mini_results_openai.csv'
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
    
    summary_file = 'mini_test_results/mini_summary_openai.csv'
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

#open ai rate limit is 3000 requests per minute