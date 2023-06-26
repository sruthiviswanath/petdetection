# Object Recognition Pipeline Documentation

This documentation provides an overview of the implementation and steps taken to complete the "Object Recognition" code challenge. The goal of this challenge is to build an object recognition pipeline using a pre-trained model and evaluate its performance on a dataset. 

## Dataset

The chosen dataset for this challenge is the "The Oxford-IIIT Pet Dataset". This dataset contains images of 37 different breeds of cats and dogs. The dataset provides images and their corresponding labels for training and testing.

## Object Detection Model

The YOLOv5 model architecture from the open-source Ultralytics was selected as the object detection model for this challenge. YOLOv5 is a state-of-the-art object detection algorithm known for its accuracy and efficiency.

## Implementation Steps

The implementation of the object recognition pipeline follows the provided instructions. The following steps were performed:

1. Dependencies Installation:
   - Install the required dependencies by running `pip install -r yolov5/requirements.txt`.

2. Data Augmentation:
   - Since yolo training has augmentation No specific data augmentation techniques were applied in this implementation. However, a sample function named `data_augmentation.py` is provided for reference.

3. Dataset Loading and Splitting:
   - The dataset was downloaded and stored in the appropriate directories.
   - The function `data_annotation.py` was used to convert the XML annotations to YOLO annotations.
   - The function `data_split.py` was used to split the dataset into training, validation, and testing sets.

4. Model Training:
   - The YOLOv5 model was trained using the provided dataset and the YOLOv5 training script.
   - A data config YAML file (`petdataset.yaml`) was created to specify the locations of the train, test, and validation images, as well as the number of classes and their names.
   - The hyperparameters for training were defined using the `hyp.scratch-med.yaml` file.
   - The YOLOv5 model (`yolov5s.yaml`) with smaller architecture was used for training.
   - The model was trained for 3 epochs with a batch size of 32.
   - The trained model weights were saved as `pet_det_26jun`.

5. Model Inference and Evaluation:
   - The trained model was tested on a separate test dataset using the `detect.py` script.
   - The model's performance was evaluated using the `val.py` script, which calculates the Average Precision for each class and mean Average Precision (mAP).
   - The evaluation results, including the confusion matrix, F1 curve, precision-recall curve, and ROC curve, were stored in the `runs/val/yolo_det2/` directory.

## Handling Bias

To handle potential biases in the dataset, the following steps were taken:
- Missing data: Some images in the dataset didn't have labels. These images were removed during the data splitting step to ensure a more balanced dataset.
- Model Training: The YOLOv5 model was trained with default hyperparameters. The `hyp.scratch-med.yaml` file with higher augmentation hyperparameters was used to reduce and delay overfitting. The hyperparameter `obj` was set to 0.7 to help reduce overfitting.

## Usage Steps

To reproduce the object recognition pipeline, follow these steps:

1. Download the dataset from [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/).
2. Clone the YOLOv5 repository by running `git clone https://github.com/ultralytics/yolov5`.
3. Install the required dependencies by running `pip install -r yolov5/requirements.txt`.
4. Copy the image and XML folders from the downloaded dataset to the current directory.
5. Convert the XML annotations to YOLO format by running `python data_annotation.py`.
6. Split the dataset into train, test, and validation sets by running `python data_split.py`.
7. Create a data config file (`petdataset.yaml`) in the `yolov5/data/` directory, specifying the locations of the train, test, and validation images, the number of classes, and their names.
8. Change to the `yolov5` directory.
9. Train the YOLOv5 model by running the command:
   ```
   python train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch-med.yaml --batch 32 --epochs 3 --data petdataset.yaml --weights yolov5s.pt --workers 24 --name pet_det
   ```
10. Perform inference on the test dataset by running the command:
    ```
    python detect.py --source ../test/images/ --weights runs/train/pet_det/weights/best.pt --conf 0.25 --name pet_det
    ```
11. Evaluate the model by running the command:
    ```
    python val.py --weights runs/train/pet_det/weights/best.pt --data petdataset.yaml --task test --name yolo_det
    ```

Please note that the above steps assume a Unix-like environment, and you may need to make adjustments depending on your specific operating system and setup.
