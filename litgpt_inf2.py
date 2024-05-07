#!/usr/bin/env python3

import requests
import json
import sys
import re
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy import stats
# Load JSON data from a file
with open('qm7_test_smi.json', 'r') as file:
    data = json.load(file)

llm_out = []
gt = []
# Iterate over each item in the JSON data
for item in data:
    prompt_text = item['instruction']  # Extract the 'instruction' key for each item

    # Make a POST request with the extracted prompt
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"prompt": prompt_text}
    )

    matches = re.findall(r'\d+\.\d+', response.json()['output'])
    # Print the output from the response
    #print(f"Prompt: {prompt_text}", flush=True)
    #gt.append(float(item['output']))
    # Check if matches were found
    if matches:
        try:
            # Attempt to convert the first match to float and append to llm_out
            result = matches[0]
            llm_out.append(float(result))
            gt.append(float(item['output']))
            #print("Successfully appended:", result)
        except ValueError as e:
            # Handle the case where conversion to float fails
            #llm_out.append(None)
            print("Error converting to float:", e, flush=True)
    else:
        print("No number found in the text.", flush=True)
    
    #print(response.json())
    #print(f"Response Output: {response.json()['output']}\n", flush=True)

print(len(gt))
print(len(llm_out))
ml = np.array(llm_out)
gt = np.array(gt)
mse = np.mean((np.subtract(gt, ml)))
error = np.subtract(gt, ml)
mad = np.mean(np.abs(np.subtract(error, mse)))
r2 = r2_score(gt, ml)
slope, intercept, r_value, p_value, std_err = stats.linregress(gt, ml)
cod = r_value**2
fig, ax = plt.subplots()
plt.scatter(gt, ml, s=25, alpha=0.5, linewidths=1, marker='o')
ax.set_aspect('equal')
#ax.set_xlim([np.min([gt, ml])-5, np.max([gt, ml])+5])
#ax.set_ylim([np.min([gt, ml])-5, np.max([gt, ml])+5])
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
line_x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
line_y = (slope*line_x) + intercept
plt.plot(line_x, line_y, color='red')
plt.annotate(r'    $r^2$ = {:.2f}'.format(cod), xy=(0.05, 0.95), xycoords='axes fraction')
ax.set_xlabel(r'QM7 Ground Truth (kcal mol$^{-1}$)')
ax.set_ylabel(r'QM7 LLM (kcal mol$^{-1}$)')
plt.tight_layout()
plt.savefig("test1.png", dpi=300)


