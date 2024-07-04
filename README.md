# Simple MLOPS workflow example

## Train the model

To train the model, run the following command:
```py
python3 src/train_model.py
```


## Start the prediction service

1. start the docker container
```sh
docker compose up
```

2. test the prediction service: create a python file `test_prediction_service.py` withe the following content:
```python
import requests

url = "http://localhost:8000/predict"

data = {
    "data": [
        [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    ]
}

response = requests.post(url, json=data)
print(response.json())
```
