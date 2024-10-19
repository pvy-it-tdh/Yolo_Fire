from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO('yolov8x.pt')

result = model.train(data='mydataset.yaml', epochs=50)



