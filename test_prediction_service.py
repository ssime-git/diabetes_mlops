import requests

url = "http://localhost:8000/predict"

data = {
    "data": [
        [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    ]
}

response = requests.post(url, json=data)
print(response.json())