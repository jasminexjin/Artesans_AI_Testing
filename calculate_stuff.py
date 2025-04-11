import math
import pandas as pd
import numpy as np
import scipy.stats as stats 
p = 0.7
N = 1000
zscore = 1.645
E = 0.05
margin_error = zscore * math.sqrt((p*(1-p))/N)
n0 = (zscore**2 * p * (1-p))/E**2
sample_size = (n0*N)/(N-1+n0)
sample_size_infinite = (zscore**2 * p * (1-p))/E**2

def mean_confidence_interval_tstats(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def mean_confidence_interval_normal(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.norm.ppf((1+confidence)/2.)
    return m, m-h, m+h

df_mixed1 = pd.read_csv('/Users/illiabilokonov/Desktop/Artesans_Storage/mini_test_results/mini_results_openai_gemini.csv')
df_mixed2 = pd.read_csv('/Users/illiabilokonov/Desktop/Artesans_Storage/mini_test_results/mini_results_gemini_openai.csv')
data_mixed_1 = df_mixed1['if_matched_correctly'].tolist()
data_mixed_2 = df_mixed2['if_matched_correctly'].tolist()

p_hat1_t, p_higher1_t, p_lower1_t = mean_confidence_interval_tstats(data_mixed_1)
p_hat2_t, p_higher2_t, p_lower2_t = mean_confidence_interval_tstats(data_mixed_2)

p_hat1_n, p_higher1_n, p_lower1_n = mean_confidence_interval_normal(data_mixed_1)
p_hat2_n, p_higher2_n, p_lower2_n = mean_confidence_interval_normal(data_mixed_2)


print(f"openai_gemini_tstats: {p_higher1_t} to {p_lower1_t}")
print(f"gemini_openai_tstats: {p_higher2_t} to {p_lower2_t}")
print(f"openai_gemini_normal: {p_higher1_n} to {p_lower1_n}")
print(f"gemini_openai_normal: {p_higher2_n} to {p_lower2_n}")

print(f"Sample size for finite population: {sample_size}")
print(f"Sample size for infinite population: {sample_size_infinite}")
print(f"Margin of error: {margin_error}")
