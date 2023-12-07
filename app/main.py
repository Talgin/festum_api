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
import faiss as fs
from pymilvus import connections, CollectionSchema, Collection
import table_structure

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

if settings.use_milvus:
    connections.connect("default", host="localhost", port="19530")
    schema = CollectionSchema(table_structure.fields, "actors_milvus is a database of IMDB, kpop and kazakh")
    milvus_collection = Collection(settings.milvus_schema, schema, consistency_level="Strong")
    # Before conducting a search or a query, you need to load the data in `hello_milvus` into memory.
    milvus_collection.load()
else:
    db_worker = PowerPost(settings.PG_CONNECTION[0], settings.PG_CONNECTION[1], settings.PG_CONNECTION[2], settings.PG_CONNECTION[3], settings.PG_CONNECTION[4])
    faiss_index = fs.read_index(settings.FAISS_INDEX_FILE, fs.IO_FLAG_ONDISK_SAME_DIR)

@app.post("/upload/images")
async def upload_images(images: List[UploadFile] = File(...)):
    file_names = {}
    for image in images:
        # grab the uploaded image
        data = await image.read()
        # Save the image to disk or process it as needed
        # You can use the `image.filename` attribute to get the original filename

        # Example: Saving the image with a unique name
        file_name = f"{uuid.uuid4()}.jpg"
        # with open(os.path.join(settings.CROPS_FOLDER, file_name), "wb") as f:
        #     f.write(contents)
        # file_names.append(file_name)

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
            # file_names.append({unique_id: res})
            file_names[unique_id] = res
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
    return {"detections": file_names}

@app.get("/detector/get_detection", status_code=200)
async def get_detection(response: Response, date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...)):
    file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg')
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'result': 'error', 'message':'No such file'}
    
@app.post("/recognition/get_photo_metadata", status_code=200)
async def get_photo_metadata(response: Response, date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...)):
    img_name = 'align_'+face_id
    file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, img_name+'.jpg')
    if os.path.exists(file_path):
        img = cv2.imread(file_path)
        feature = None
        # using either gpu or cpu to get feature - the use_cpu is in settings.py
        if settings.use_cpu:
            feature = recognizer.cpu_get_feature(img, unique_id+'_'+img_name)
        else:
            feature = recognizer.get_feature(img, unique_id+'_'+img_name, 0)

        if settings.use_postgres:
            # get top one from postgres database of people
            db_result = db_worker.get_top_one_from_face_db(feature)
            if len(db_result) > 0:
                ids, distances = utils.calculate_cosine_distance(db_result, feature, settings.RECOGNITION_THRESHOLD)
                if ids is not None:
                    l_name = db_worker.search_from_persons(ids)
                    return {
                            'result': 'success',
                            'message': {
                                        'id': ids,
                                        'name': l_name,
                                        'similarity': round(distances * 100, 2)
                                    }
                        }
            else:
                response.status_code = status.HTTP_409_CONFLICT
                return {'result': 'error', 'message': 'No IDs found'}
        elif settings.use_milvus:
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10},
            }
            result = milvus_collection.search([feature], "person_feature", search_params, limit=3, output_fields=["person_name", "person_surname", "person_middlename"])
            result_dict = {
                            'result': 'success',
                            'message': {
                                        'distance': round(result[0][0].distance, 2),
                                        'person_id': result[0][0].id,
                                        'surname': result[0][0].entity.get("person_surname"),
                                        'firstname': result[0][0].entity.get("person_name"),
                                        'secondname': result[0][0].entity.get("person_middlename")
                                    }
                        }
            return result_dict
        else:
            if faiss_index.ntotal > 0:
                distances, indexes = db_worker.search_from_gbdfl_faiss_top_n(faiss_index, feature, 1)
            else:
                return {'result': 'error', 'message': 'FAISS index is empty.'}
        
            if indexes is not None:
                ids = tuple(list(map(str,indexes[0])))
                # ids = str(list(indexes[0]))[1:-1]
                print("IDs", ids)
                from_ud_gr = db_worker.get_info_from_stars_database(ids)
                # with_zeros = []
                # str_ids = list(map(str, indexes[0]))
                # for i in str_ids:
                #     while len(i) < 9:
                #         i = "0" + i
                #     with_zeros.append(i)
                # print('ZEROs ADDED:', with_zeros)
                # from_ud_gr = db_worker.get_blob_info_from_database(tuple(with_zeros))
                print('FROM DATABASE:', from_ud_gr)
                if from_ud_gr is not None:
                    scores_val = dict(zip(list(ids),list(distances[0])))
                    print('DICTIONARY:', scores_val)
                    for i in range(len(from_ud_gr)):
                        print(from_ud_gr[i][0])
                        dist = scores_val[str(from_ud_gr[i][0])]
                        unique_id = from_ud_gr[i][0]
                        # gr_code = from_ud_gr[i][1]
                        surname = from_ud_gr[i][1]
                        firstname = from_ud_gr[i][2]
                        if from_ud_gr[i][3] is None:
                            secondname = ''
                        else:
                            secondname = from_ud_gr[i][3]
                        # fio = surname +' '+ firstname +' '+secondname
                        result_dict = {
                                        'result': 'success',
                                        'message': {
                                                    'distance': round(dist*100, 2),
                                                    'person_id': unique_id,
                                                    'surname': surname,
                                                    'firstname': firstname,
                                                    'secondname': secondname
                                                }
                                    }
                    return result_dict
            else:
                return {'result': 'error', 'message': 'ud_gr is empty'}
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'result': 'error', 'message': 'No such file. Please, check unique_id, face_id or date.'}