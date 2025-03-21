import logging

import os

import io
import PIL
import json
import uvicorn
import requests
import moviepy
import aiofiles

import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, HTTPException,Depends,Form

from typing import Optional
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.requests import Request

import moviepy.editor as mp
from keras.models import load_model
from google import genai
from google.genai import types
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError  # Import the correct exception class
from starlette.requests import Request
from pydantic import ValidationError
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# os.environ['TF_KERAS'] = '1'
# Set the maximum upload size to 100 megabytes
# max_upload_size = 100 * 1024 * 1024  # 100 MB
# Set environment variable for Google credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './sqy-prod.json'

origins = [
    "https://ai.propvr.tech",
    "http://ai.propvr.tech",
    "https://ai.propvr.tech/classify",
    "http://ai.propvr.tech/classify",
    "https://uatapp.smartagent.ae",
    "https://app.smartagent.ae",
    "http://app.smartagent.ae/*",
    "http://localhost:8081/*",
    "http://uatapp.smartagent.ae/*"
]

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
)

# Configure middleware to handle larger file uploads
middleware = [
    Middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # Replace with your actual allowed hosts
    ),
    Middleware(
        CORSMiddleware,
        allow_origins=origins,  # Use the provided list of origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ),
]

app = FastAPI(middleware=middleware,MAX_UPLOAD_SIZE = 100 * 1024 * 1024)

# app.config.REQUEST_MAX_SIZE = 100 * 1024 * 1024   # 10MB in bytes
# Load the model for image Classification
try:
    model = load_model('classificaton_model_1oct.h5', compile=False)
except Exception as e:
    raise HTTPException(status_code=500, detail="Error loading the image classification model")

# Load the model for watermark-detection
try:
    # detect_model = load_model('detect_watermark.h5', compile=False)
    detect_model = load_model('keras_model_6nov.h5', compile=False)
except Exception as e:
    raise HTTPException(status_code=500, detail="Error loading the image classification model")


# Define the image size
image_size = (224, 224)


@app.get("/")
async def root():
    return "Server is up!"


def predict_img(image: Image.Image):
        try:
            labels = [
                'Bathroom', 'Bedroom', 'Living Room', 'Exterior View', 'Kitchen', 'Garden', 'Plot', 'Room',
                'Swimming Pool', 'Gym', 'Parking', 'Map Location', 'Balcony', 'Floor Plan', 'Furnished Amenities',
                'Building Lobby', 'Team Area', 'Staircase', 'Master Plan','Plot Area', 'false'
            ]
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = ImageOps.fit(image, image_size)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            
            # Ensure that normalized_image_array is 3D (height x width x channels)
            if normalized_image_array.shape[-1] == 3:
                img_array_final = normalized_image_array
            
            else:
                # Handle the case where the image has a different number of channels (e.g., grayscale)
                # You may need to adapt this part based on your requirements.
                img_array_final = normalized_image_array[:, :, 0:3]

            img_array_final = normalized_image_array[:, :, :3]
            data[0] = img_array_final
            result = model.predict(data)
            arr_sorted = -np.sort(-result, axis=1)
            top_five = arr_sorted[:, :5]
            top_five_array = result.argsort()
            top_five_array1 = top_five_array.tolist()
            top1 = top_five_array1[0][-1]
            top2 = top_five_array1[0][-2]
            top3 = top_five_array1[0][-3]
            top4 = top_five_array1[0][-4]
            top5 = top_five_array1[0][-5]
            index_max = np.argmax(result)

            # Check if 'false' is in the top 2 predictions
            is_false_present = labels[top1] == "false" or labels[top2] == "false"

            # Update the top prediction to 'false' if 'false' exists in top 2
            top_prediction_label = "false" if is_false_present else str(labels[index_max])
            top_prediction_confidence = str(top_five[0][1]) if is_false_present else str(result[0][index_max])

                
            prediction_dict = {
                "response": {
                    "solutions": {
                        "re_roomtype_eu_v2": {
                            "predictions": [
                                {
                                    "confidence": str(top_five[0][0]),
                                    "label": str(labels[top1])
                                },
                                {
                                    "confidence": str(top_five[0][1]),
                                    "label": str(labels[top2])
                                }
         
                            ],
                            
                            "top_prediction": {
                                "confidence": top_prediction_confidence,
                                "label": top_prediction_label
                            }
                        }
                    }
                }
            }

            return prediction_dict
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error predicting image")



def check_watermark(image: Image.Image):
        try:
            labels = [
                'Watermark', 'Non-Watermark'
            ]
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = ImageOps.fit(image, image_size)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            
            # Ensure that normalized_image_array is 3D (height x width x channels)
            if normalized_image_array.shape[-1] == 3:
                img_array_final = normalized_image_array
            
            else:
                # Handle the case where the image has a different number of channels (e.g., grayscale)
                # You may need to adapt this part based on your requirements.
                img_array_final = normalized_image_array[:, :, 0:3]

            img_array_final = normalized_image_array[:, :, :3]
            data[0] = img_array_final
            result = detect_model.predict(data)
            arr_sorted = -np.sort(-result, axis=1)
            top_five = arr_sorted[:, :5]
            top_five_array = result.argsort()
            top_five_array1 = top_five_array.tolist()
            top1 = top_five_array1[0][-1]
            index_max = np.argmax(result)

            if str(labels[top1]) == "Non-Watermark":
                (labels[top1]) = "0"
            else :
                (labels[top1]) = "1"
                
            prediction_dict = {
                "response": {
                    "solutions": {
                        "re_roomtype_eu_v2": {
                            "predictions": [
                                {
                                    "confidence": str(top_five[0][0]),
                                    "label": str(labels[top1])
                                },
                               
                            ],
                            "top_prediction": {
                                "confidence": str(result[0][index_max]),
                                "label": str(labels[index_max])
                            }
                        }
                    }
                }
            }

            return prediction_dict
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error predicting image")
        

@app.get("/predict_from_url")
async def predict_image(image_url: str):

    image_error = {
                "response": {
                    "solutions": {
                        "re_roomtype_eu_v2": {
                            "predictions": [
                                {
                                    "confidence": "",
                                    "label": "-1"
                                },
                               
                            ],
                            "top_prediction": {
                                "confidence": "",
                                    "label": "-1"
                            }
                        }
                    }
                }
            }

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = io.BytesIO(response.content)
        input_image = Image.open(image_bytes).convert("RGB")
        input_image.save("input_img.jpg")
        prediction_result = predict_img(input_image)
        os.remove("input_img.jpg")
        return prediction_result
    
    except:
        return image_error


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    '''This function uploads an image from your system'''
    image_error = {
                "response": {
                    "solutions": {
                        "re_roomtype_eu_v2": {
                            "predictions": [
                                {
                                    "confidence": "",
                                    "label": "-1"
                                },
                               
                            ],
                            "top_prediction": {
                                "confidence": "",
                                    "label": "-1"
                            }
                        }
                    }
                }
            }
    try:
        contents = await file.read()
        image_bytes = Image.open(BytesIO(contents))
        image_bytes.save("input_img.jpg")
        prediction_result = predict_img(image_bytes)
        os.remove("input_img.jpg")
        return prediction_result
    # except PIL.UnidentifiedImageError as e:
    #     raise HTTPException(status_code=400, detail="Invalid image format")
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail="Error processing image")
    except:
        return image_error


@app.get("/detect_from_url")
async def detection_image(image_url: str):
        image_error = {
                "response": {
                    "solutions": {
                        "re_roomtype_eu_v2": {
                            "predictions": [
                                {
                                    "confidence": "",
                                    "label": "-1"
                                },
                               
                            ],
                            "top_prediction": {
                                "confidence": "",
                                    "label": "-1"
                            }
                        }
                    }
                }
            }
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_bytes = io.BytesIO(response.content)
            input_image = Image.open(image_bytes).convert("RGB")
            input_image.save("input_img.jpg")
            prediction_result = check_watermark(input_image)
            os.remove("input_img.jpg")
            return prediction_result
        except:
            return image_error


@app.post("/detect-file")
async def detection_from_file(file: UploadFile = File(...)):
    '''This function uploads an image from your system'''
    image_error = {
                "response": {
                    "solutions": {
                        "re_roomtype_eu_v2": {
                            "predictions": [
                                {
                                    "confidence": "",
                                    "label": "-1"
                                },
                               
                            ],
                            "top_prediction": {
                                "confidence": "",
                                    "label": "-1"
                            }
                        }
                    }
                }
            }
    try:
        contents = await file.read()
        image_bytes = Image.open(BytesIO(contents)).convert("RGB")
        image_bytes.save("input_img.jpg")
        prediction_result = check_watermark(image_bytes)
        return prediction_result
    except:
        return image_error


class VideoClassificationResult(BaseModel):
    is_real_estate: bool


@app.post("/classify_video")
async def classify_video(video: UploadFile = File(...)):
    

    try:
        
        logging.info(f"Received video for classification: {video.filename}")
        
        # Temporary file path to save the uploaded video
        temp_video_path = "temp_video.mp4"
        
        # Save the uploaded video to a temporary file
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(video.file.read())

        # Ensure the file is properly closed before proceeding
        video.file.close()

        # Load the image classification model
        model = load_model("video_classification.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()

        # Create a temporary directory to store frames as image files
        # temp_dir = "temp_frames"
        # os.mkdir(temp_dir, exist_ok=True)

        # Extract frames from the video and save them as image files
        frames = []
        real_estate = []
        non_real_estate = []

        clip = mp.VideoFileClip(temp_video_path)
        for i, frame in enumerate(clip.iter_frames(fps=1)):
            image = Image.fromarray(frame)

            # Resize and normalize the image for classification
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predict the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Categorize frames as real estate or non-real estate
            if class_name[2:] == "Non real estate\n":
                non_real_estate.append(class_name[2:])
            else:
                real_estate.append(class_name[2:])

        # Determine the result based on frame classification
        is_real_estate = len(real_estate) > len(non_real_estate)

        # # Clean up temporary directory and video file
        # for frame_file in os.listdir(temp_dir):
        #     os.remove(os.path.join(temp_dir, frame_file))
        # os.rmdir(temp_dir)
        # Close the VideoFileClip object
        clip.close()
        os.remove(temp_video_path)
        if len(real_estate) > len(non_real_estate):
            response = "1"
            print("real_estate = ",response)
        else:
            response = "0"
            print("non_real_estate = ",response)
            
        # Logging the result
        logging.info(f"Video {video.filename} classified as {'real estate' if is_real_estate else 'non-real estate'}")
        
        return VideoClassificationResult(is_real_estate=is_real_estate)
    
    except Exception as e:
        # Log any errors
        logging.error(f"Error processing video {video.filename}: {str(e)}")
        
        return {"error": str(e)}
    


async def process_image(client, image_data: bytes):
    # Save uploaded file temporarily
    temp_file = "temp_image.jpeg"
    async with aiofiles.open(temp_file, 'wb') as f:
        await f.write(image_data)
    uploaded_file = client.files.upload(file=temp_file)
    file_uri = uploaded_file.uri
    mime_type = uploaded_file.mime_type
    # Clean up temporary file
    os.remove(temp_file)

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=file_uri,
                    mime_type=mime_type,
                ),
                types.Part.from_text(text="""Classify this image and detect watermark on image"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "classification": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    enum=[
                         'Bathroom', 'Bedroom', 'Living Room', 'Exterior View', 'Kitchen', 'Garden', 'Plot Area', 'Room',
                'Swimming Pool', 'Gym', 'Parking', 'Map Location', 'Balcony', 'Floor Plan', 'Furnished Amenities',
                'Building Lobby', 'Team Area', 'Staircase', 'Master Plan', 'false'
                    ],
                ),
            },
        ),

        system_instruction=[
            types.Part.from_text(text="""
                You are an image classifier for images shown in real estate listings.
                - If image not related to real-estate then return 'false'.
                - If the image has a transparent watermark from '99acres' or 'housing.com', return 'false'."""),
        ],
    )
    result = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        result += chunk.text

    
    # Parse the JSON response and extract just the classification value
    parsed_result = json.loads(result)
    return parsed_result["classification"]

@app.post("/classify-image-file/")
async def classify_image_file(
    image_file: UploadFile = File(...)
):
    """
    Classify an image by uploading a file
    - image_file: The image file to classify
    """
    try:
        client = genai.Client(
            api_key="AIzaSyBeNEyjpgf8tX2AQJSunqSIOBfPGr08DS8",
        )

        image_data = await image_file.read()
        classification = await process_image(client, image_data=image_data)
        
        return JSONResponse(content={"result": classification})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/classify-image-url/")
async def classify_image_url(
    image_url: str = Form(...)
):
    """
    Classify an image by providing a URL
    - image_url: URL of the image to classify
    """
    try:
        client = genai.Client(
            api_key="AIzaSyBeNEyjpgf8tX2AQJSunqSIOBfPGr08DS8",
        )

        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()
        image_data = response.content
        
        classification = await process_image(client, image_data=image_data)
        
        return JSONResponse(content={"result": classification})

    except requests.RequestException as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to download image from URL: {str(e)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )