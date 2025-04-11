import pandas as pd
import streamlit as st
from ocr_plugin import extract_text_from_image
from streamlit import session_state
import os
import openpyxl

EXCEL_FILE = 'Testing/full_data_cleaned.xlsx'
existing_data = pd.read_excel(EXCEL_FILE, engine= 'openpyxl')
existing_df = pd.DataFrame(existing_data)

CSV_FILE = 'Testing/ocr_text.csv'


if not os.path.exists(CSV_FILE):
    ocr_df = pd.DataFrame(columns=['ocr_text', 'actual_index', 'actual_name', 'actual_expiration_date'])
else:
    ocr_csv = pd.read_csv(CSV_FILE)
    ocr_df = pd.DataFrame(ocr_csv)



st.title("Create Test")
image = st.camera_input("scan the product")

if 'ocr_df' not in st.session_state:
    st.session_state.ocr_df = ocr_df



if image:
    ocr_text = extract_text_from_image(image)
    st.write(ocr_text)
    actual_index = st.number_input("Enter the index of the matched product", step=1)
    macthed_name = existing_df.iloc[actual_index, 2]
    actual_index -=2
    macthed_expiration_date = existing_df.iloc[actual_index, 3]
    data = ([{'ocr_text': ocr_text, 
              'actual_index': actual_index, 
              'actual_name': macthed_name, 
              'actual_expiration_date': macthed_expiration_date}])
    session_state.ocr_df= pd.DataFrame(data)
    if st.button("Save", key="save_button"):
        ocr_df = pd.concat([ocr_df, session_state.ocr_df], ignore_index=True)
        ocr_df.to_csv(CSV_FILE, index=False)
    
    edited_df = st.data_editor(ocr_df, num_rows="dynamic")
    if st.button("Save Edited Data", key="save_edited_data_button"):
        ocr_df = edited_df
        ocr_df.to_csv(CSV_FILE, index=False)
        
def get_name_date_index(df, index):
    return df.iloc[index, 2], df.iloc[index, 3], df.iloc[index, 0]

def get_ocr_csv():
    results = pd.read_csv(CSV_FILE)
    return results

def get_ocr_df():
    return ocr_df









