# 
# Download Dataset(https://www.crowdhuman.org/)

Convert to yolov8 format

python convert_crowdhuman_to_yolov8.py annotation_train.odgt yolov8_train

python convert_crowdhuman_to_yolov8.py annotation_val.odgt yolov8_val

# Train Model

pip install 

!yolo detect train data=/content/content/CrowdHuman-3/data.yaml model=/content/drive/MyDrive/dl_project/train/weights/last.pt epochs=100 imgsz=640 project="/content/drive/MyDrive/dl_project" resume=True

# Track

!yolo track model=/content/drive/MyDrive/dl_project/train/weights/last.pt source=... 

