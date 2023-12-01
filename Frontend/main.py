from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("D:/plant-disease-project/saved_models/1.h5")

CLASS_NAMES = ['Beans__Angular_Leaf_Spot',
 'Beans__Bean_Rust',
 'Beans__Healthy',
 'Blackgram__Anthracnose',
 'Blackgram__Healthy',
 'Blackgram__Yellow__Mosaic',
 'Corn__Diseased',
 'Corn__Healthy',
 'Cotton__Curl__Virus',
 'Cotton__Fussarium__Wilt',
 'Cotton__Healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___healthy',
 'Mango_Sooty__Mould',
 'Mango__Bacterial__Canker',
 'Mango__Healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Rice_Brown_Spot',
 'Rice_False_Smut',
 'Rice__Healthy',
 'Rose__Black__Spot',
 'Rose__Downy__Mildew',
 'Rose__Healthy',
 'Tea__Anthracnose',
 'Tea__Healthy',
 'Tea__White__Spot',
 'Tea__brown__blight',
 'Tea__red_leaf _spot',
 'Tomato__Early_blight',
 'Tomato__Late_blight',
 'Tomato__healthy']



@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)