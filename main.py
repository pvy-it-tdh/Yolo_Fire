from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load a pretrained YOLO model
model = YOLO('yolov8x.pt')

result = model.train(data='mydataset.yaml', epochs=3, device='cpu')



