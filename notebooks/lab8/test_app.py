import requests

iris_data = {
    "sepal_length": 1,
    "sepal_width": 3.5,
    "petal_length": 2,
    "petal_width": 0.2
}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=iris_data)
print(response.json())