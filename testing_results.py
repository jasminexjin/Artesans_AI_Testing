import pandas as pd

'''
- ocr_text.csv : a list of ocr_text from the image. Contains ocr_text, actual_name, actual_expiration_date, actual_index (Needs a quick frontend scan image and to add the actual_index)
- ocr_results.pd (Optional to csv): a df of Products after the first API call. Contains ocr_text, name, expiration_date, index,quantity, actual_name, actual_expiration_date, actual_index
- matching_products.pd: a df of the matching results after the second API call. Contains ocr_text, name, expiration_date, index,quantity, matched_name, matched_expiration_date, matched_index
    - for each ocr_text, it will matched to a dict of matched_name, matched_expiration_date, matched_index

- final_results.pd (Optional to csv): a df of the final results. Contains ocr_text, name, expiration_date, index,quantity, if_matched_correctly (True/False)
    - if_matched_correctly is True if one of the matched_index is the same as the actual index

- final_results_summary.csv: a csv of the final results. Contains ai_model, accuracy, time_first_api_call, time_second_api_call, time_total, cost
'''

claude_money_before = 0.48
openai_money_before = 5-4.44
mistral_money_before = 0.00
gemini_money_before = 0.37


claude_money_after = 2.96
openai_money_after = 5-2.81
mistral_money_after = 0.00
gemini_money_after = 0.37

openai_mixed_first = 5-2.75
openai_mixed_second = 5-1.23
claude_money_used = claude_money_after - claude_money_before
openai_money_used = openai_money_after - openai_money_before
mistral_money_used = mistral_money_after - mistral_money_before
gemini_money_used = gemini_money_after - gemini_money_before


total_mixed_1 = openai_mixed_first - openai_money_used
total_mixed_2 = openai_mixed_second - openai_mixed_first

claude_df = pd.read_csv('/Users/illiabilokonov/Desktop/Artesans_Storage/mini_test_results/mini_summary_claude.csv')
claude_df['cost'] = claude_money_used
openai_df = pd.read_csv('/Users/illiabilokonov/Desktop/Artesans_Storage/mini_test_results/mini_summary_openai.csv')
openai_df['cost'] = openai_money_used
mistral_df = pd.read_csv('/Users/illiabilokonov/Desktop/Artesans_Storage/mini_test_results/mini_summary_mistral.csv')
mistral_df['cost'] = mistral_money_used
gemini_df = pd.read_csv('/Users/illiabilokonov/Desktop/Artesans_Storage/mini_test_results/mini_summary_gemini.csv')
gemini_df['cost'] = gemini_money_used

combo1_df = pd.read_csv('/Users/illiabilokonov/Desktop/Artesans_Storage/mini_test_results/mini_summary_openai_gemini.csv')
combo2_df = pd.read_csv('/Users/illiabilokonov/Desktop/Artesans_Storage/mini_test_results/mini_summary_gemini_openai.csv')
combo1_df['cost'] = total_mixed_1
combo2_df['cost'] = total_mixed_2
final_df = pd.concat([claude_df, openai_df, mistral_df, gemini_df, combo1_df, combo2_df])

final_df.to_excel('/Users/illiabilokonov/Desktop/Artesans_Storage/mini_test_results/mini_summary_final.xlsx', index=False)










