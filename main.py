# from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from typing import List

import os
import sys
import cv2
import numpy as np
import io
from datetime import datetime
import uuid

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from recognition import Recognition
from detection import Detector
import settings
import utils
from db.powerpostgre import PowerPost
import time
import faiss as fs

app = FastAPI()

def create_triton_client(server, protocol, verbose, async_set, cpu_server):
    triton_client = None
    if settings.use_cpu:
        try:
            if protocol == "grpc":
                # Create gRPC client for communicating with the server
                triton_client = grpcclient.InferenceServerClient(url=cpu_server, verbose=verbose)
                triton_client.is_server_live()
            else:
                # Specify large enough concurrency to handle the number of requests.
                concurrency = 20 if async_set else 1
                triton_client = httpclient.InferenceServerClient(url=cpu_server, verbose=verbose, concurrency=concurrency)
            print('Connected to CPU Model Server')
        except Exception as e:
            print("CPU client creation failed: " + str(e))
            # sys.exit(1)
    else:
        try:
            if protocol == "grpc":
                # Create gRPC client for communicating with the server
                triton_client = grpcclient.InferenceServerClient(url=server, verbose=verbose)
                triton_client.is_server_live()
            else:
                # Specify large enough concurrency to handle the number of requests.
                concurrency = 20 if async_set else 1
                triton_client = httpclient.InferenceServerClient(url=server, verbose=verbose, concurrency=concurrency)
            print('Connected to GPU Model Server')
        except Exception as e:
            print("GPU client creation failed: " + str(e))
            triton_client = None
            # sys.exit(1)
    return triton_client

triton_client = create_triton_client(settings.TRITON_SERVER_SETTINGS[0], settings.TRITON_SERVER_SETTINGS[1], settings.TRITON_SERVER_SETTINGS[2], settings.TRITON_SERVER_SETTINGS[3], settings.TRITON_SERVER_SETTINGS[4])
detector = Detector(triton_client, settings.use_cpu, settings.DETECTOR_SETTINGS[0], settings.DETECTOR_SETTINGS[1], settings.DETECTOR_SETTINGS[2], settings.DETECTOR_SETTINGS[3], settings.DETECTOR_SETTINGS[4], settings.DETECTOR_SETTINGS[5], settings.DETECTOR_SETTINGS[6])
recognizer = Recognition(triton_client, settings.use_cpu, settings.RECOGNITION_SETTINGS[0], settings.RECOGNITION_SETTINGS[1], settings.RECOGNITION_SETTINGS[2], settings.RECOGNITION_SETTINGS[3], settings.RECOGNITION_SETTINGS[4])
if settings.use_postgres:
    db_worker = PowerPost(settings.PG_CONNECTION[0], settings.PG_CONNECTION[1], settings.PG_CONNECTION[2], settings.PG_CONNECTION[3], settings.PG_CONNECTION[4])
else:
    faiss_index = fs.read_index(settings.FAISS_INDEX_FILE, fs.IO_FLAG_ONDISK_SAME_DIR)

@app.post("/upload/images")
async def upload_images(images: List[UploadFile] = File(...)):
    file_names = []
    for image in images:
        contents = await image.read()
        # Save the image to disk or process it as needed
        # You can use the `image.filename` attribute to get the original filename

        # Example: Saving the image with a unique name
        file_name = f"image_{uuid.uuid4()}.jpg"
        with open(os.path.join(settings.CROPS_FOLDER, file_name), "wb") as f:
            f.write(contents)
        file_names.append(file_name)

    return {"file_names": file_names}