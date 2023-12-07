import json
import cv2
import os
import tqdm

def convert_crowdhuman_to_yolov8(annotation_file, output_dir):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for annotation in tqdm.tqdm(annotations):
        image_id = annotation['ID']
        image_path = os.path.join(os.path.dirname(annotation_file), 'images', image_id)
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        output_file = os.path.join(output_dir, image_id + '.txt')
        with open(output_file, 'w') as f:
            for gtbox in annotation['gtboxes']:
                if gtbox['tag'] == 'person':
                    x1, y1, w, h = gtbox['vbox']
                    x_center = (x1 + w) / 2
                    y_center = (y1 + h) / 2
                    normalized_width = w / image_width
                    normalized_height = h / image_height

                    # Convert to YOLOv8 format
                    yolo_line = f"{gtbox['tag']} {x_center} {y_center} {normalized_width} {normalized_height}"
                    f.write(yolo_line + '\n')