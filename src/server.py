import uvicorn
from fastapi import FastAPI

from inference import classify

app = FastAPI()


@app.get('/')
async def root():
    return {'Howdy': 'partner!'}


@app.get("/inference/{tweet}")
async def inference(tweet: str):
    tweet += ' ->'
    classification = classify([tweet])
    return {"inference": classification}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

"""
TODO
Impement automated unit-tests that call the API
Implement a GUI with gradio and/or streamlit 
Use Hugging Face endpoints
Implement the inference for OpenAI's GPT too
"""