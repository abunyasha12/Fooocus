import time
from base64 import b64encode
from io import BytesIO

import uvicorn
from fastapi import FastAPI
from PIL import Image

import modules.le_headless_async_worker as worker
from modules.le_headless_async_worker import Txt2imgRequest, Txt2imgResponse
from modules.sdxl_styles import styles

app = FastAPI()

@app.get("/focus_api/styles/")
async def get_styles():
    return styles

@app.post("/focus_api/txt2img/")
async def txt2img(parameters: Txt2imgRequest):
    img_list: list[bytes] = []
    # args = [val for val in parameters.model_dump().values()]
    worker.buffer.append(parameters)
    finished = False
    buffered = BytesIO()

    while not finished:
        time.sleep(0.01)
        if len(worker.outputs) > 0:
            flag, product = worker.outputs.pop(0)
            if flag == 'results':
                finished = True

    for img in product:
        Image.fromarray(img).save(buffered, format="PNG")
        img_list.append(b64encode(buffered.getvalue()))

    return Txt2imgResponse(images=tuple(img_list))

uvicorn.run(app, port=7759)

