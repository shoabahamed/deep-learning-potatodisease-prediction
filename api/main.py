from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import keras

app = FastAPI()
MODEL = keras.models.load_model('../models/2')
class_names = ['Early_blight', 'Late_blight', 'healthy']


@app.get('/ping')
async def ping():
    return 'Server running'


def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    print()
    return image


@app.post('/predict')
async def predict(
    file: UploadFile
):
    img = read_file_as_image(await file.read())
    batch_img = np.expand_dims(img, axis=0)
    predictions = MODEL.predict(batch_img)[0]
    prediction = np.argmax(predictions)
    confidence = (np.max(predictions)) * 100
    class_name = class_names[prediction]

    return f'The image is of {class_name} with a confidence of {round(confidence, 2)}'


if __name__ == "__main__":
     uvicorn.run(app, host='localhost', port=8000)





# what uploaded file in the async function returns is binary string. We convert it to byte string
# thought the command file.read(). To convert byte string to byte array which pillow can read
# we use io.BytesIO. Then we open with pillow and convert it numpy array