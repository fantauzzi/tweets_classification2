import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from inference import classify
from fastapi.testclient import TestClient

app = FastAPI()
test_client = TestClient(app)


class Tweets(BaseModel):
    tweets: list[str]


@app.get('/')
async def root():
    return {'Howdy': 'partner!'}


@app.post("/inferences/")
async def inferences(tweets: Tweets):
    preprocessed_tweets = [item + ' ->' for item in tweets.tweets]
    classification = classify(preprocessed_tweets)
    return {"inference": classification}


@app.get("/inference/{tweet}")
async def inference(tweet: str):
    tweet += ' ->'
    classification = classify([tweet])
    return {"inference": classification}


def main():
    uvicorn.run(app, host='127.0.0.1', port=8000)


if __name__ == '__main__':
    main()

"""
TODO
Try unit-test with WebSocket sessions, see https://www.starlette.io/testclient/
Implement a GUI with gradio and/or streamlit 
Use Hugging Face endpoints
Implement the inference for OpenAI's GPT too
Do deployment in a docker container
Implement authentication in the API
"""
