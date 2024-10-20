from ultralytics import YOLO
from PIL import Image

# Load a pretrained YOLO model
model = YOLO(r'D:\Python\YoloV8\runs\detect\train13\weights\best.pt')
result = model('D:\Python\YoloV8\chay.webp')
for r in result:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[...,::-1])
    im.show()

