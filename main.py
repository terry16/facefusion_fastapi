#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from server import process_face_swap
from facefusion import logger

app = FastAPI()

class FaceSwapRequest(BaseModel):
    srcUrl: str
    targetUrl: str

@app.post("/face_swap")
async def face_swap(request: FaceSwapRequest):
    try:
        output_path = process_face_swap(request.srcUrl, request.targetUrl)
        return {"output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    logger.init('debug')
    uvicorn.run(app, host="0.0.0.0", port=8666)