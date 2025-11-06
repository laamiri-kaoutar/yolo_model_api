from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# Load your trained YOLO model
model = YOLO("./model/best.pt")  # replace with your trained weights

app = FastAPI(title="YOLO Image API")

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    results = model.predict(source=image, conf=0.25, verbose=False)

    # Use Ultralytics built-in plotting (like Streamlit)
    result_img = results[0].plot()  # returns BGR numpy array
    result_img = Image.fromarray(result_img[..., ::-1])  # BGR → RGB

    buf = BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")