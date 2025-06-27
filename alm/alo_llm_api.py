import json
from fastapi import APIRouter, HTTPException, File, UploadFile, Request, Query, Form
from fastapi import File, UploadFile
from pydantic import BaseModel
import os
import inspect
import sys
from typing import Any, Dict
import yaml
import aiofiles
from pathlib import Path
from fastapi.responses import JSONResponse
##
def load_yaml_file(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

# 현재 작업 디렉토리 얻기
current_directory = os.getcwd()
# 현재 작업 디렉토리와 'config.yaml'을 합쳐서 절대 경로 생성
config_path = os.path.join(current_directory, 'config.yaml')
config = load_yaml_file(config_path)

class UpdateAPI:
    def __init__(self):
        self.router = APIRouter(prefix="/api/v1")

    # def setup_routes(self):

    #     @self.router.post("/uploadfile/")
    #     async def upload_file(file: UploadFile = File(...), id: str = Form(...)):
    #         try:
    #             environment = os.getenv('env_flag')
    #             if not environment:
    #                 print("Local Test 환경")
    #                 try:
    #                     directory = f".workspace/{id}"
    #                     file_location = f"{directory}/{file.filename}"

    #                     # 폴더가 없으면 생성
    #                     if not os.path.exists(directory):
    #                         os.makedirs(directory)

    #                     async with aiofiles.open(file_location, "wb") as out_file:
    #                         content = await file.read()
    #                         await out_file.write(content)

    #                     return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    #                 except Exception as e:
    #                     return {"error": str(e)}

    #             elif environment == 'AWS':
    #                 try: 
    #                     import boto3
    #                     ald_s3_client = boto3.client('s3')
    #                     ald_s3_path = os.getenv('ald_s3_path')
    #                     if not ald_s3_path:
    #                         return JSONResponse(status_code=400, content={"error": "s3_path가 제공되지 않았습니다."})

    #                     if ald_s3_path.startswith("S3://"):
    #                         ald_s3_path = ald_s3_path[len("S3://"):]
    #                     path_parts = ald_s3_path.split("/", 1)

    #                     if len(path_parts) != 2:
    #                         return JSONResponse(status_code=400, content={"error": "s3_path 형식이 올바르지 않습니다. '버킷이름/경로' 형식이어야 합니다."})
                        
    #                     bucket_name, object_key = path_parts
    #                     ald_stream_id = os.getenv('stram_id')
    #                     ald_stream_history = os.getenv('stream_history')
    #                     object_key = f"{object_key}/{ald_stream_id}/{ald_stream_history}/{id}/"
    #                     ald_s3_client.upload_file(file.file, bucket_name, object_key)
    #                     return JSONResponse(status_code=200, content={"message": f"File uploaded to {ald_s3_client} successfully"})
                    
    #                 except Exception as e:
    #                     return JSONResponse(status_code=500, content={"error": str(e)})
                    
    #             elif environment == 'GCP':
    #                 print("GCP TBD")
    #             elif environment == 'LOCAL':
    #                 print("LOCAL TBD")
    #             else:
    #                 print('Not Supported')

    #         except Exception as e:
    #             return {"error": str(e)}

    #     @self.router.post("/downloadfile/")
    #     async def download_file(file: UploadFile = File(...), id: str = Form(...)):
    #         try:
    #             environment = os.getenv('env_flag')
    #             if not environment:
    #                 print("Local Test 환경")
    #                 directory = f".workspace/{id}"
    #                 file_location = f"{directory}/{file.filename}"

    #                 # 폴더가 없으면 생성
    #                 if not os.path.exists(directory):
    #                     os.makedirs(directory)

    #                 if not os.path.exists(file_location):
    #                     raise HTTPException(status_code=404, detail="File not found")
                    
    #                 return file_location

    #             elif environment == 'AWS':
    #                 try: 
    #                     import boto3
    #                     ald_s3_client = boto3.client('s3')

    #                     download_path = f".workspace/{id}"
    #                     # 폴더가 없으면 생성합니다.
    #                     if not os.path.exists(download_path):
    #                         os.makedirs(download_path)

    #                     ald_s3_path = os.getenv('ald_s3_path')
    #                     if not ald_s3_path:
    #                         return JSONResponse(status_code=400, content={"error": "s3_path가 제공되지 않았습니다."})

    #                     if ald_s3_path.startswith("S3://"):
    #                         ald_s3_path = ald_s3_path[len("S3://"):]
    #                     path_parts = ald_s3_path.split("/", 1)

    #                     if len(path_parts) != 2:
    #                         return JSONResponse(status_code=400, content={"error": "s3_path 형식이 올바르지 않습니다. '버킷이름/경로' 형식이어야 합니다."})
                        
    #                     bucket_name, object_key = path_parts
    #                     ald_stream_id = os.getenv('stram_id')
    #                     ald_stream_history = os.getenv('stream_history')

    #                     folder_name = f"{object_key}/{ald_stream_id}/{ald_stream_history}/{id}/"

    #                     objects = ald_s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)


    #                     if 'Contents' not in objects:
    #                         print(f"No objects found in {bucket_name}/{folder_name}")
    #                         return

    #                     for obj in objects['Contents']:
    #                         object_key = obj['Key']
    #                         file_name = os.path.basename(object_key)
    #                         local_file_path = os.path.join(download_path, file_name)

    #                         try:
    #                             ald_s3_client.download_file(bucket_name, object_key, local_file_path)
    #                             print(f'{object_key} has been downloaded to {local_file_path}')
    #                         except Exception as e:
    #                             print(f'Error downloading {object_key}: {e}')

    #                     return JSONResponse(status_code=200, content={"message": f"File uploaded to {ald_s3_client} successfully"})
                    
    #                 except Exception as e:
    #                     return JSONResponse(status_code=500, content={"error": str(e)})
                    
    #             elif environment == 'GCP':
    #                 print("GCP TBD")
    #             elif environment == 'LOCAL':
    #                 print("LOCAL TBD")
    #             else:
    #                 print('Not Supported')

    #         except Exception as e:
    #             return {"error": str(e)}


    #     @self.router.get("/log_files")
    #     def get_logs_in_workspace_logs_folder():
    #         workspace_log_dir = Path(".workspace/logs")
    #         if not workspace_log_dir.exists():
    #             raise HTTPException(status_code=404, detail="Workspace 'logs' directory not found.")

    #         current_dir_files = [f.name for f in workspace_log_dir.iterdir() if f.is_file()]
    #         return {"folders": current_dir_files}
        
    #     @self.router.get("/read_logs")
    #     async def read_log_file(file_path: str):
    #         log_file_path = os.path.join(".workspace/logs", file_path)
    #         log_file_path = Path(log_file_path)
    #         if not log_file_path.exists():
    #             raise HTTPException(status_code=404, detail=f"Log file not found: {file_path}")

    #         try:
    #             with log_file_path.open("r", encoding="utf-8") as file:
    #                 content = file.read()
    #             return {"content": content}
    #         except Exception as e:
    #             raise HTTPException(status_code=500, detail=f"An error occurred while reading the log file: {str(e)}")

    # def get_router(self):
    #     """라우터 반환 메서드 추가"""
    #     return self.router