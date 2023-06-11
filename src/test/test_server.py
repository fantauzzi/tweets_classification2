"""
Note: the working directory to run the unit-test should be src, not src/test, otherwise there will be errors
"""
import sys
from pathlib import Path

parent = (Path(__file__).parent / '..').resolve()
sys.path.append(str(parent))

import json

from server import test_client


def test_root():
    res = test_client.get('http://127.0.0.1:8000')
    assert res.status_code == 200
    content_decoded = res.content.decode('utf8')
    content_json = json.loads(content_decoded)
    assert content_json['Howdy'] == 'partner!'


def test_inference():
    tweet = 'i dont blame it all to them and im not angry at them infact i feel fairly sympathetic for them'
    res = test_client.get(f'http://127.0.0.1:8000/inference/{tweet}')
    assert res.status_code == 200

