"""
It's tests.
Spins up a cog server and hits it with the HTTP API if you're local; runs through the python client if you're not
"""


import base64
import os
import subprocess
import sys
import pytest
import requests
import replicate
from functools import partial

ENV = os.getenv('TEST_ENV', 'local')
LOCAL_ENDPOINT = "http://localhost:5000/predictions"
MODEL = os.getenv('REPLICATE_MODEL', 'replicate/hello_world')

def local_http_generation(model_endpoint, **kwargs):
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


def replicate_run(model, version, **kwargs):
    output = replicate.run(
        f"{model}:{version}",
        input=kwargs)
    return output


@pytest.fixture(scope="session")
def inference_func():
    if ENV == 'local':
        return partial(local_http_generation, LOCAL_ENDPOINT)
    model = replicate.models.get(MODEL)
    version = model.versions.list()[0]
    return partial(replicate_run, MODEL, version)


@pytest.fixture(scope="session", autouse=True)
def service():
    if ENV == 'local':
        # starts local server if we're running things locally
        build_command = 'cog build -t test-model'.split()
        subprocess.check_call(build_command)

        run_command = 'docker run -d -p 5000:5000 --gpus all test-model'.split()
        process = subprocess.Popen(run_command, stdout=sys.stdout, stderr=sys.stderr)

        yield
        process.terminate()
        process.wait()
    else:
        yield



def test_upscale(inference_func):
    
