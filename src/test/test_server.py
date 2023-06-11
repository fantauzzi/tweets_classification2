"""
Note: the working directory to run the unit-test should be src, not src/test, otherwise there will be errors
"""
import sys
from pathlib import Path

parent = (Path(__file__).parent / '..').resolve()
sys.path.append(str(parent))

import json

from server import test_client
from datasets import load_dataset


def test_root():
    res = test_client.get('http://127.0.0.1:8000')
    assert res.status_code == 200
    content_decoded = res.content.decode('utf8')
    content_json = json.loads(content_decoded)
    assert content_json['Howdy'] == 'partner!'


def check_inference_result(inference_text: str) -> int:
    res_obj = json.loads(inference_text)
    inference = res_obj.get('inference')
    assert inference is not None
    for item in inference:
        label = item.get('label')
        assert label is not None
        label_int = int(label[-1])
        assert label_int >= 0
        assert label_int <= 5
        score = item.get('score')
        assert score is not None
        assert score >= 0
        assert score <= 1
    return len(inference)


def test_inference():
    tweet = 'i dont blame it all to them and im not angry at them infact i feel fairly sympathetic for them'
    res = test_client.get(f'http://127.0.0.1:8000/inference/{tweet}')
    assert res.status_code == 200
    count = check_inference_result(res.text)
    assert count == 1


def test_inferences():
    emotion = load_dataset('emotion')
    n_samples = 64
    tweets = emotion['test']['text'][:n_samples]
    data = {"tweets": tweets}
    data_str = json.dumps(data)
    res = test_client.post(url='http://127.0.0.1:8000/inferences/', data=data_str)
    assert res.status_code == 200
    count = check_inference_result(res.text)
    assert count == n_samples
