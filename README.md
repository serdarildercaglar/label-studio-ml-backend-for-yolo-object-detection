---

# Label Studio YOLO Object Detection ML Backend

This repository provides a custom ML backend for Label Studio to use YOLO object detection model for pre-annotations while annotating data. The ML backend is implemented using Flask and performs predictions via the YOLO model.

## Prerequisites

Ensure you have the following software installed:
- Python (>=3.6)
- Label Studio
- Flask
- ultralytics (for YOLO)

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/serdarildercaglar/label-studio-ml-backend-for-yolo-object-detection
   cd label-studio-ml-backend-for-yolo-object-detection
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the model path and labels**:

   You need to manually set the path to your YOLO model weights and the label configurations in the `YOLOv10Model` class. Open the script and locate the following sections and update them accordingly:

   ```python
   model_path = '/path/to/your/model/weights/best.pt'  # Change this to your model weights path
   self.labels_in_config = { 0: "class1", 1: "class2" }  # Change this to your actual labels and class IDs
   ```

## Running the Backend

To start the Flask server for the ML backend, run:

```bash
python app.py
```

The server will start running on `http://0.0.0.0:9090`.

## Integrating with Label Studio

1. **Create a project** in Label Studio.

2. **Go to Project Settings**:
   - Click on your project to open it.
   - Navigate to the `Settings` tab.

3. **Configure the ML Backend**:
   - Select the `Model` tab.
   - Enter the ML backend URL as `http://your-ip-address:9090`.

## Usage

Once configured, when you annotate data in Label Studio, the YOLO model will provide pre-annotations based on your custom backend.

## Endpoints

The server exposes the following endpoints:
- `/predict`: For making predictions on tasks sent by Label Studio.
- `/health`: For checking the health status of the server.
- `/setup`: Used for model setup.
- `/webhook`: For handling webhook requests from Label Studio.



## Troubleshooting

If you encounter any issues, ensure that:
- The Flask server is running and accessible from the Label Studio instance.
- The `model_path` and `labels_in_config` are correctly set to match your model and labels.

## Contributing

Feel free to raise issues or submit pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License.

---
