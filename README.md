# Simple MLOPS workflow example

Below are the minimal steps to create a simple MLOPS workflow. Inspiration from [this article](https://towardsdatascience.com/simple-model-retraining-automation-via-github-actions-b0f61d5c869c).

The code and the logic have refactored with best practices in mind.

## Create the environment

Create a virtual environment and install the requirements:
```sh
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Model exploration

Explore the notebook `notebooks/diabetes_notebook.ipynb` to see the data and the model exploration. Make sure to use the `venv` environment.

## Write the python training scripts and run it (to generate the model)

To train the model, run the following command:
```py
python3 src/train_model.py
```

CHALLENGE : To go a bit further you could write the ingestion and preprocessing scripts that will get raw data and prepare it for training.

## Write the Dockerfile and docker-compose.yml

Now with a trained model, we can write the Dockerfile and docker-compose.yml to create a container that will serve the model.

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

## writhe CI/CD pipeline
Since your ML service is ready and you have a prediction service, you can write a CI/CD pipeline to automate the training and deployment of the model. You can use GitHub Actions to do this. Create a `.github/workflows/main.yml`.