"""
It's tests.
Spins up a cog and runs locally if you're local; hits staged model in github
"""


import base64
import os
import pytest
import requests

@pytest.fixture(scope="session")
def model_endpoint():
    return os.getenv('TEST_URL', "http://localhost:5000/predictions")  # Default to local URL

@pytest.fixture(scope="session", autouse=True)
def start_local_service_if_needed(model_endpoint):
    if "localhost" in model_endpoint:
        # TODO: start cog model if we're running locally.
        yield
    else:
        yield

def run_generation(model_endpoint, **kwargs):
    # TODO - probably some API token shenanigans to do here.
    response = requests.post(model_endpoint, json={"input": kwargs})
    data = response.json()

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except Exception as e:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        raise e


# TODO: define a few tests and run them, locally and otherwise.
