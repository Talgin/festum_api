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

import utils

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
        # grab the uploaded image
        data = await image.read()
        # Save the image to disk or process it as needed
        # You can use the `image.filename` attribute to get the original filename

        # Example: Saving the image with a unique name
        file_name = f"{uuid.uuid4()}.jpg"
        # with open(os.path.join(settings.CROPS_FOLDER, file_name), "wb") as f:
        #     f.write(contents)
        file_names.append(file_name)

        # compare each uploaded face with database - get top 1 from database for the person
        # then we can take this top 1 and say that the person looks like the person from database
        # we have to set some threshold for this

        #print('data_type:', type(data))
        # unique_id = str(round(time.time() * 1000000))
        # name = image.filename
        img = np.asarray(bytearray(data), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # print('Image shape:', img.shape)
        
        if settings.use_cpu:
            faces, landmarks = detector.cpu_detect(img, file_name, settings.DETECTION_THRESHOLD)
        else:
            faces, landmarks = detector.detect(img, file_name, 0, settings.DETECTION_THRESHOLD)
        # print(faces.shape[0])
        
        if faces.shape[0] > 0:
            res, unique_id = utils.process_faces(img, faces, landmarks)
            print(res)
            '''
            match_list = {}
            feature_list = []
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int)
                # Getting the size of head rectangle
                height_y = box[3] - box[1]
                width_x = box[2] - box[0]
                # Calculating cropping area
                if landmarks is not None and height_y > 40:
                    center_y = box[1] + ((box[3] - box[1])/2)
                    center_x = box[0] + ((box[2] - box[0])/2)
                    rect_y = int(center_y - height_y/2)
                    rect_x = int(center_x - width_x/2)
                    # Get face alignment
                    landmark5 = landmarks[i].astype(np.int)
                    aligned = align_img(image, landmark5)
                    # Get 512-d embedding from aligned image
                    feature = recognizer.get_feature(aligned, file_name, 0)
                    feature_list.append(feature)
                else:
                    return {'result': False, 'message': 'Face not detected or sharp angle'}

            if faiss_index.ntotal > 0 and feature_list:
                # distances, indexes = db_worker.search_from_blacklist_faiss_top_1(faiss_index, feature, 1, threshold)
                distances, indexes = db_worker.search_faiss_multiple_vectors(faiss_index, feature_list, threshold)
            else:
                message = {'result': False, 'message': 'Faiss index is empty.', 'data': None}
                return message
            if indexes:
                print(indexes, distances)
                message = {'result': True, 'message': 'Success', 'name': str(indexes), 'faces': faces.shape[0]}
                return message
            else:
                message = {'result': False, 'message': 'Not found', 'name': "", 'faces': faces.shape[0]}
                return message
            '''
        # compare each uploaded face with database - get top 1 from database for the person
        # then we can take this top 1 and say that the person looks like the person from database
        # we have to set some threshold for this
    return {"file_names": file_names}