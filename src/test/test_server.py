import sys
from pathlib import Path  # if you haven't already done so

parent = (Path(__file__).parent / '..').resolve()
sys.path.append(str(parent))

import requests
import json
import multiprocessing
from time import sleep

import pytest
import uvicorn

from server import app


def run_server(host, port):
    uvicorn.run(app, host=host, port=port)


@pytest.fixture(autouse=True, scope="session")
def start_server():
    p = multiprocessing.Process(target=run_server, args=('127.0.0.1', 8000))
    p.start()
    res = None
    retries = 30
    while (res is None or res.status_code != 200) and retries > 0:
        sleep(1)
        res = requests.get('http://127.0.0.1:8000')
        retries -= 1
    yield
    p.terminate()


def test_root(start_server):
    res = requests.get('http://127.0.0.1:8000')
    assert res.status_code == 200
    content_decoded = res.content.decode('utf8')
    content_json = json.loads(content_decoded)
    assert content_json['Howdy'] == 'partner!'
