from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "yolo.h5"
input_path = "a.jpg"
output_path = "newimage.jpg"

detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"], " : ", eachItem["percentage_probability"])
