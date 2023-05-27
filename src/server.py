from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@app.get('/')
def root():
    return {'Howdy': 'partner!'}


class Item(BaseModel):
    name: str
    price: float
    tags: list[str] = []


@app.post("/items/")
def create_item(item: Item):
    return item