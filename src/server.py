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
    uvicorn.run(app, host='0.0.0.0', port=8000)
