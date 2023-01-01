import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
from networks.layers import AdaIN, AdaptiveAttention
from retinaface.models import *
from utils.options import FaceDancerOptions
from utils.swap_func import run_inference
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import os, shutil

logging.getLogger().setLevel(logging.ERROR)

opt = FaceDancerOptions().parse()

if len(tf.config.list_physical_devices('GPU')) != 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[opt.device_id], 'GPU')

print('\nInitializing FaceDancer...')
RetinaFace = load_model(opt.retina_path, compile=False, custom_objects = {"FPN": FPN,
                                                                          "SSH": SSH,
                                                                          "BboxHead": BboxHead,
                                                                          "LandmarkHead": LandmarkHead,
                                                                          "ClassHead": ClassHead})

ArcFace = load_model(opt.arcface_path, compile=False)

G = load_model(opt.facedancer_path, compile=False, custom_objects={"AdaIN": AdaIN,
                                                                   "AdaptiveAttention": AdaptiveAttention,
                                                                   "InstanceNormalization": InstanceNormalization})
G.summary()


app = FastAPI()
    
@app.get("/")
def test_api():
    return {"Hello": "World"}

@app.post("/faceswap")
def generate(content_image: UploadFile = File(...), target_image: UploadFile = File(...)):
    os.makedirs('./results', exist_ok=True)
    os.system('rm -rf ./results/*')
    
    # save cnotent image file
    input_image_file_path1 = f"./source_image/{content_image.filename}"
    with open(input_image_file_path1, "wb") as buffer:
        shutil.copyfileobj(content_image.file, buffer)
    
    # save target image file
    input_image_file_path2 = f"./swap_image/{target_image.filename}"
    with open(input_image_file_path2, "wb") as buffer:
        shutil.copyfileobj(target_image.file, buffer)
    
    print('\nProcessing: {}'.format(opt.img_path))
    run_inference(opt, input_image_file_path1, input_image_file_path2, RetinaFace, ArcFace, G, opt.img_output)
    print('\nDone! {}'.format(opt.img_output))
    
uvicorn.run(app, host='0.0.0.0', port=8000)
