# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:21:17 2024

@author: pruth
"""

import requests
import time
import json

# Assuming 'a' is your object, serialize it to JSON format
# This is necessary if 'a' is not natively serializable by requests
a = {"key": "value"}  # example object
json_data = json.dumps(a)

# URL of the Flask API
url = "http://127.0.0.1:5000/api/data"

# Start time
start_time = time.time()

# Sending the POST request to the Flask API
response = requests.post(url, json=json_data)

# End time
end_time = time.time()

# Calculate duration
duration = end_time - start_time
print(f"Request sent and response received in {duration:.2f} seconds.")

# Output the response from the server
print("Response from server:", response.json())
