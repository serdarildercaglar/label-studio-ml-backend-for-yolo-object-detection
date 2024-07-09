from flask import Flask, request, jsonify
from label_studio_ml.model import LabelStudioMLBase
# from ultralytics import YOLOv10 ## for 8.2.,1
from ultralytics import YOLO
import numpy as np
import os
import logging
from typing import List, Dict, Optional
from label_studio_ml.utils import get_image_size, get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_data_dir, get_local_path

app = Flask(__name__)

logger = logging.getLogger(__name__)

class YOLOv10Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOv10Model, self).__init__(**kwargs)
        model_path = '/mnt/sdb1/indivitech/konu-anlatım-kitabı-kesme/lise/model-training/runs/detect/train7/weights/best.pt'
        # self.model = YOLOv10(model_path) ## for 8.2.31
        self.model = YOLO(model_path)
        
        # default image upload folder
        upload_dir = os.path.join(get_data_dir(), "media", "upload")
        self.image_dir = kwargs.get('image_dir', upload_dir)
        logger.debug(f"{self.__class__.__name__} reads images from {self.image_dir}")

        # Manually set the label config parameters
        self.from_name = "label"  # Replace with your actual from_name
        self.to_name = "image"  # Replace with your actual to_name
        self.value = "image"  # Replace with your actual value
        self.labels_in_config = {0: "q", 1: "t"}  # Replace with your actual labels and class IDs
        self.score_threshold = float(os.environ.get("SCORE_THRESHOLD", 0.5))

        # Initialize model version
        self.model_version = "0.0.1"

    def setup(self):
        """Configure any parameters of your model here"""
        self.model_version = "0.0.1"

    def _get_image_url(self, task: Dict) -> str:
        image_url = task["data"].get(self.value) or task["data"].get(DATA_UNDEFINED_NAME)
        return image_url

    def predict(self, tasks: List[Dict], **kwargs):
        predictions = []
        for task in tasks:
            prediction = self.predict_one_task(task)
            predictions.append(prediction)
        return {"results": predictions}  # Return a dictionary with "results" key

    def predict_one_task(self, task: Dict):
        image_url = self._get_image_url(task)
        image_path = get_local_path(image_url, task_id=task.get('id'))
        results = self.model.predict(image_path)
        
        temp_result = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        
        for result in results:
            boxes = result.boxes  # Bounding box bilgilerini içerir
            for box in boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()  # veya box.xywh, box.xyxyn gibi farklı formatlar kullanılabilir
                confidence = box.conf.item()
                class_id = box.cls.item()
                
                # bbox score is too low
                if confidence < self.score_threshold:
                    continue
                
                x = x_min / img_width * 100
                y = y_min / img_height * 100
                width = (x_max - x_min) / img_width * 100
                height = (y_max - y_min) / img_height * 100
                
                label = self.labels_in_config.get(class_id, "unknown")  # class_id'ye göre label belirle
                
                temp_result.append({
                    "original_width": img_width,
                    "original_height": img_height,
                    "image_rotation": 0,
                    "value": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "rotation": 0,
                        "rectanglelabels": [label]  
                    },
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "origin": "manual",
                    "score": confidence
                })
                all_scores.append(confidence)
        
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        
        # Debug: Print the prediction results
        print("Prediction results:", temp_result)
        
        if not temp_result:
            print("No predictions were made.")
        
        return {"result": temp_result, "score": avg_score, "model_version": self.model_version}

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.model_version
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.model_version = 'my_new_model_version'
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.model_version}')

        print('fit() completed successfully.')

model = YOLOv10Model()

@app.route('/predict', methods=['POST'])
def predict():
    tasks = request.json['tasks']
    print(tasks)
    results = model.predict(tasks)
    
    return jsonify(results)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({"status": "ok"})

@app.route("/webhook", methods=["POST"])
def webhook():
    # Handle the webhook request from Label Studio
    data = request.json
    print("Webhook received:", data)
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
