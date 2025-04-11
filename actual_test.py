import pandas as pd
from openai_ai import get_product_details_openai, get_matching_products_openai
from claude_ai import get_product_details_claude, get_matching_products_claude
from gemini_ai import get_product_details_gemini, get_matching_products_gemini
from mistral_ai import get_product_details_mistral, get_matching_products_mistral
import time
import json
import os

os.makedirs('test_results', exist_ok=True)

ocr_csv = pd.read_csv('Testing/ocr_text.csv')
ocr_df = pd.DataFrame(ocr_csv)
existing_products_csv = pd.read_excel('Testing/full_data_cleaned.xlsx')
existing_products_df = pd.DataFrame(existing_products_csv)
matching_products_df_openai = ocr_df.copy()
matching_products_df_claude = ocr_df.copy()
matching_products_df_mistral = ocr_df.copy()
matching_products_df_gemini = ocr_df.copy()

# Initialize matched_dict columns with object dtype
matching_products_df_openai['matched_dict'] = None
matching_products_df_openai['matched_dict'] = matching_products_df_openai['matched_dict'].astype('object')
matching_products_df_claude['matched_dict'] = None
matching_products_df_claude['matched_dict'] = matching_products_df_claude['matched_dict'].astype('object')
matching_products_df_mistral['matched_dict'] = None
matching_products_df_mistral['matched_dict'] = matching_products_df_mistral['matched_dict'].astype('object')
matching_products_df_gemini['matched_dict'] = None
matching_products_df_gemini['matched_dict'] = matching_products_df_gemini['matched_dict'].astype('object')

# Create result dataframes for each model
final_results_df_openai = ocr_df.copy()
final_results_df_claude = ocr_df.copy()
final_results_df_mistral = ocr_df.copy()
final_results_df_gemini = ocr_df.copy()

# Initialize results columns
for df in [final_results_df_openai, final_results_df_claude, final_results_df_mistral, final_results_df_gemini]:
    df['matched_correctly'] = 0

product_details_df = pd.DataFrame(columns=['ocr_text', 'name', 'expiration_date', 'actual_name', 'actual_expiration_date', 'actual_index'])
final_results_summary_df = pd.DataFrame(columns=['ai_model', 'accuracy', 'time_first_api_call', 'time_second_api_call', 'time_total', 'cost'])

first_api_call_time_openai = 0
first_api_call_time_claude = 0
first_api_call_time_mistral = 0
first_api_call_time_gemini = 0

def convert_json_to_pd(json_data):
    data_dict = json.loads(json_data)
    df = pd.DataFrame(data_dict['products'])
    return df

def convert_json_to_dict(json_data):
    data_dict = json.loads(json_data)
    return data_dict

def get_accuracy(ocr_df: pd.DataFrame, matching_products_df: pd.DataFrame):
    count_correct = 0
    final_results_df = ocr_df.copy()
    final_results_df['if_matched_correctly'] = 0
    for i, row in ocr_df.iterrows():
        ocr_text = row['ocr_text']
        actual_index = row['actual_index']
        matched_row = matching_products_df.loc[matching_products_df['ocr_text'] == ocr_text]
        
        if len(matched_row) == 0 or matched_row['matched_dict'].iloc[0] is None:
            continue
            
        matched_dict = matched_row['matched_dict'].iloc[0]
        matched_indices = [p.get('index') for p in matched_dict.get('products', [])]
        
        if actual_index in matched_indices:
            final_results_df.loc[i, 'if_matched_correctly'] = 1
            count_correct += 1
    
    total_accuracy = count_correct / len(ocr_df) if len(ocr_df) > 0 else 0
    return total_accuracy, final_results_df
    
def first_api_openai(ocr_text: str):
    start_time_openai = time.time()
    product_details_openai = get_product_details_openai(ocr_text)
    end_time_openai = time.time()
    time_openai = end_time_openai - start_time_openai
    product_details_openai_df = convert_json_to_pd(product_details_openai)
    return product_details_openai_df, time_openai
def second_api_openai(product_name: str, expiration_date: str, top_n: int, existing_products: pd.DataFrame):
    #openai
    start_time_openai = time.time()
    matching_products_openai = get_matching_products_openai(product_name, expiration_date, top_n, existing_products)
    end_time_openai = time.time()
    time_openai = end_time_openai - start_time_openai
    matched_dict_openai = convert_json_to_dict(matching_products_openai)
    return matched_dict_openai, time_openai


def first_api_claude(ocr_text: str):
    start_time_claude = time.time()
    product_details_claude = get_product_details_claude(ocr_text)
    end_time_claude = time.time()
    time_claude = end_time_claude - start_time_claude
    product_details_claude_df = convert_json_to_pd(product_details_claude)
    return product_details_claude_df, time_claude
def second_api_claude(product_name: str, expiration_date: str, top_n: int, existing_products: pd.DataFrame):
    start_time_claude = time.time()
    matching_products_claude = get_matching_products_claude(product_name, expiration_date, top_n, existing_products)
    end_time_claude = time.time()
    time_claude = end_time_claude - start_time_claude
    matched_dict_claude = convert_json_to_dict(matching_products_claude)
    return matched_dict_claude, time_claude 


def first_api_mistral(ocr_text: str):
    start_time_mistral = time.time()
    product_details_mistral = get_product_details_mistral(ocr_text)
    end_time_mistral = time.time()
    time_mistral = end_time_mistral - start_time_mistral
    product_details_mistral_df = convert_json_to_pd(product_details_mistral)
    return product_details_mistral_df, time_mistral 
def second_api_mistral(product_name: str, expiration_date: str, top_n: int, existing_products: pd.DataFrame): 
    start_time_mistral = time.time()
    matching_products_mistral = get_matching_products_mistral(product_name, expiration_date, top_n, existing_products)
    end_time_mistral = time.time()
    time_mistral = end_time_mistral - start_time_mistral
    matched_dict_mistral = convert_json_to_dict(matching_products_mistral)
    return matched_dict_mistral, time_mistral


def first_api_gemini(ocr_text: str):
    start_time_gemini = time.time()
    product_details_gemini = get_product_details_gemini(ocr_text)
    end_time_gemini = time.time()
    time_gemini = end_time_gemini - start_time_gemini
    product_details_gemini_df = convert_json_to_pd(product_details_gemini)
    return product_details_gemini_df, time_gemini
def second_api_gemini(product_name: str, expiration_date: str, top_n: int, existing_products: pd.DataFrame):

    start_time_gemini = time.time()
    matching_products_gemini = get_matching_products_gemini(product_name, expiration_date, top_n, existing_products)
    end_time_gemini = time.time()
    time_gemini = end_time_gemini - start_time_gemini
    matched_dict_gemini = convert_json_to_dict(matching_products_gemini)
    return matched_dict_gemini, time_gemini


def get_product_details(ocr_text: str):
    #openai
    start_time_openai = time.time()
    product_details_openai = get_product_details_openai(ocr_text)
    end_time_openai = time.time()
    time_openai = end_time_openai - start_time_openai
    product_details_openai_df = convert_json_to_pd(product_details_openai)

    #claude
    start_time_claude = time.time()
    product_details_claude = get_product_details_claude(ocr_text)
    end_time_claude = time.time()
    time_claude = end_time_claude - start_time_claude
    product_details_claude_df = convert_json_to_pd(product_details_claude)

    #mistral
    start_time_mistral = time.time()
    product_details_mistral = get_product_details_mistral(ocr_text)
    end_time_mistral = time.time()
    time_mistral = end_time_mistral - start_time_mistral
    product_details_mistral_df = convert_json_to_pd(product_details_mistral)

    #gemini
    start_time_gemini = time.time()
    product_details_gemini = get_product_details_gemini(ocr_text)
    end_time_gemini = time.time()
    time_gemini = end_time_gemini - start_time_gemini
    product_details_gemini_df = convert_json_to_pd(product_details_gemini)


    return product_details_openai_df, product_details_claude_df, product_details_mistral_df, product_details_gemini_df, time_openai, time_claude, time_mistral, time_gemini
def get_matching_products(product_name: str, expiration_date: str, top_n: int, existing_products: pd.DataFrame):
    #openai
    start_time_openai = time.time()
    matching_products_openai = get_matching_products_openai(product_name, expiration_date, top_n, existing_products)
    end_time_openai = time.time()
    time_openai = end_time_openai - start_time_openai
    matched_dict_openai = convert_json_to_dict(matching_products_openai)

    #claude
    start_time_claude = time.time()
    matching_products_claude = get_matching_products_claude(product_name, expiration_date, top_n, existing_products)
    end_time_claude = time.time()
    time_claude = end_time_claude - start_time_claude
    matched_dict_claude = convert_json_to_dict(matching_products_claude)

    #mistral
    start_time_mistral = time.time()
    matching_products_mistral = get_matching_products_mistral(product_name, expiration_date, top_n, existing_products)
    end_time_mistral = time.time()
    time_mistral = end_time_mistral - start_time_mistral
    matched_dict_mistral = convert_json_to_dict(matching_products_mistral)

    #gemini
    start_time_gemini = time.time()
    matching_products_gemini = get_matching_products_gemini(product_name, expiration_date, top_n, existing_products)
    end_time_gemini = time.time()
    time_gemini = end_time_gemini - start_time_gemini
    matched_dict_gemini = convert_json_to_dict(matching_products_gemini)

    return matched_dict_openai, matched_dict_claude, matched_dict_mistral, matched_dict_gemini, time_openai, time_claude, time_mistral, time_gemini
    
    
    
    
    
    
    
    
    
    
    return matching_products


#openai
def openai_test(ocr_df: pd.DataFrame, results_df: pd.DataFrame):
    top_n = 3
    total_first_time_openai = 0
    total_second_time_openai = 0
    count_correct = 0
    
    for i, row in ocr_df.iterrows():
        ocr_text = row['ocr_text']
        actual_index = row['actual_index']
        
        # First API call
        product_details_openai_df, first_time_openai = first_api_openai(ocr_text)
        product_name_openai = product_details_openai_df['name'].values[0]
        product_expiration_date_openai = product_details_openai_df['expiration_date'].values[0]
        
        # Second API call
        matching_dict_openai, second_time_openai = second_api_openai(product_name_openai, product_expiration_date_openai, top_n, existing_products_df)
        
        # Track time
        total_first_time_openai += first_time_openai
        total_second_time_openai += second_time_openai
        
        # Check if actual index is in matched products
        matched_indices = [p.get('index') for p in matching_dict_openai.get('products', [])]
        if actual_index in matched_indices:
            results_df.loc[i, 'matched_correctly'] = 1
            count_correct += 1
    
    # Calculate accuracy
    accuracy = count_correct / len(ocr_df) if len(ocr_df) > 0 else 0
    
    return total_first_time_openai, total_second_time_openai, accuracy, results_df

#claude
def claude_test(ocr_df: pd.DataFrame, results_df: pd.DataFrame):
    top_n = 3
    total_first_time_claude = 0
    total_second_time_claude = 0
    count_correct = 0
    
    for i, row in ocr_df.iterrows():
        ocr_text = row['ocr_text']
        actual_index = row['actual_index']
        
        # First API call
        product_details_claude_df, first_time_claude = first_api_claude(ocr_text)
        product_name_claude = product_details_claude_df['name'].values[0]
        product_expiration_date_claude = product_details_claude_df['expiration_date'].values[0]
        
        # Second API call
        matching_dict_claude, second_time_claude = second_api_claude(product_name_claude, product_expiration_date_claude, top_n, existing_products_df)
        
        # Track time
        total_first_time_claude += first_time_claude
        total_second_time_claude += second_time_claude
        
        # Check if actual index is in matched products
        matched_indices = [p.get('index') for p in matching_dict_claude.get('products', [])]
        if actual_index in matched_indices:
            results_df.loc[i, 'matched_correctly'] = 1
            count_correct += 1
    
    # Calculate accuracy
    accuracy = count_correct / len(ocr_df) if len(ocr_df) > 0 else 0
    
    return total_first_time_claude, total_second_time_claude, accuracy, results_df

#mistral
def mistral_test(ocr_df: pd.DataFrame, results_df: pd.DataFrame):
    top_n = 3
    total_first_time_mistral = 0
    total_second_time_mistral = 0
    count_correct = 0
    
    for i, row in ocr_df.iterrows():
        ocr_text = row['ocr_text']
        actual_index = row['actual_index']
        
        # First API call
        product_details_mistral_df, first_time_mistral = first_api_mistral(ocr_text)
        product_name_mistral = product_details_mistral_df['name'].values[0]
        product_expiration_date_mistral = product_details_mistral_df['expiration_date'].values[0]
        
        # Second API call
        matching_dict_mistral, second_time_mistral = second_api_mistral(product_name_mistral, product_expiration_date_mistral, top_n, existing_products_df)
        
        # Track time
        total_first_time_mistral += first_time_mistral
        total_second_time_mistral += second_time_mistral
        
        # Check if actual index is in matched products
        matched_indices = [p.get('index') for p in matching_dict_mistral.get('products', [])]
        if actual_index in matched_indices:
            results_df.loc[i, 'matched_correctly'] = 1
            count_correct += 1
    
    # Calculate accuracy
    accuracy = count_correct / len(ocr_df) if len(ocr_df) > 0 else 0
    
    return total_first_time_mistral, total_second_time_mistral, accuracy, results_df

#gemini
def gemini_test(ocr_df: pd.DataFrame, results_df: pd.DataFrame):
    top_n = 3
    total_first_time_gemini = 0
    total_second_time_gemini = 0
    count_correct = 0
    
    for i, row in ocr_df.iterrows():
        ocr_text = row['ocr_text']
        actual_index = row['actual_index']
        
        # First API call
        product_details_gemini_df, first_time_gemini = first_api_gemini(ocr_text)
        product_name_gemini = product_details_gemini_df['name'].values[0]
        product_expiration_date_gemini = product_details_gemini_df['expiration_date'].values[0]
        
        # Second API call
        matching_dict_gemini, second_time_gemini = second_api_gemini(product_name_gemini, product_expiration_date_gemini, top_n, existing_products_df)
        
        # Track time
        total_first_time_gemini += first_time_gemini
        total_second_time_gemini += second_time_gemini
        
        # Check if actual index is in matched products
        matched_indices = [p.get('index') for p in matching_dict_gemini.get('products', [])]
        if actual_index in matched_indices:
            results_df.loc[i, 'matched_correctly'] = 1
            count_correct += 1
    
    # Calculate accuracy
    accuracy = count_correct / len(ocr_df) if len(ocr_df) > 0 else 0
    
    return total_first_time_gemini, total_second_time_gemini, accuracy, results_df
    

# Execute tests
total_first_time_claude, total_second_time_claude, total_accuracy_claude, final_results_df_claude = claude_test(ocr_df, final_results_df_claude)
total_first_time_openai, total_second_time_openai, total_accuracy_openai, final_results_df_openai = openai_test(ocr_df, final_results_df_openai)
total_first_time_mistral, total_second_time_mistral, total_accuracy_mistral, final_results_df_mistral = mistral_test(ocr_df, final_results_df_mistral)
total_first_time_gemini, total_second_time_gemini, total_accuracy_gemini, final_results_df_gemini = gemini_test(ocr_df, final_results_df_gemini)

# Add model name to result dataframes
final_results_df_claude['ai_model'] = 'claude'
final_results_df_openai['ai_model'] = 'openai'
final_results_df_mistral['ai_model'] = 'mistral'
final_results_df_gemini['ai_model'] = 'gemini'

# Create summary dataframe
final_results_summary_df = pd.DataFrame({
    'ai_model': ['claude', 'openai', 'mistral', 'gemini'],
    'accuracy': [total_accuracy_claude, total_accuracy_openai, total_accuracy_mistral, total_accuracy_gemini],
    'time_first_api_call': [total_first_time_claude, total_first_time_openai, total_first_time_mistral, total_first_time_gemini],
    'time_second_api_call': [total_second_time_claude, total_second_time_openai, total_second_time_mistral, total_second_time_gemini],
    'time_total': [total_first_time_claude + total_second_time_claude, 
                  total_first_time_openai + total_second_time_openai,
                  total_first_time_mistral + total_second_time_mistral,
                  total_first_time_gemini + total_second_time_gemini],
    'cost': ['', '', '', '']
})

# Save results
final_results_summary_df.to_csv('Testing/final_results_summary.csv', index=False)
final_results_df_claude.to_csv('Testing/final_results_claude.csv', index=False)
final_results_df_openai.to_csv('Testing/final_results_openai.csv', index=False)
final_results_df_mistral.to_csv('Testing/final_results_mistral.csv', index=False)
final_results_df_gemini.to_csv('Testing/final_results_gemini.csv', index=False)



