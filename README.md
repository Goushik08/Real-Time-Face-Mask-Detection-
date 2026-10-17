# Real-Time-Face-Mask-Detection-
Real-time face mask detection using TensorFlow, MobileNetV2 and OpenCV. Detects faces from a webcam and classifies Mask / No Mask.
Real-Time Face Mask Detection

A deep learning project that detects faces from a webcam feed and classifies whether each person is wearing a mask. It uses a two-stage pipeline: an SSD face detector finds faces, and a fine-tuned MobileNetV2 classifier decides Mask or No Mask.
How It Works
Face detection: OpenCV's DNN module runs a pretrained SSD face detector (ResNet-10 backbone, Caffe model) on each frame.
Mask classification: Each detected face is cropped, resized to 224x224, preprocessed, and passed to a MobileNetV2-based classifier trained with TensorFlow/Keras.
Output: A green box labeled "Mask" or a red box labeled "No Mask" is drawn on the live video.
Tech Stack
Python
TensorFlow / Keras (MobileNetV2, transfer learning)
OpenCV (DNN face detection, video capture, drawing)
NumPy, imutils
Jupyter Notebook
Facemask/
├── dataset/                      # Training images (not included in this repo)
│   ├── with_mask/
│   └── without_mask/
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── train_mask_detector.py        # Trains the mask classifier
├── detect_mask_image.py          # Detects masks in a single image
├── detect_mask_video.py          # Detects masks in a live video stream
├── mask_detector.model           # Trained classifier (HDF5 format)
├── requirements.txt
└── README.md
Dataset

The dataset is not included in this repository because of its size. Add your own with_mask and without_mask images, or use a public face mask dataset such as the one referenced in the original tutorial. Please follow the license of whichever dataset you use.

What I Learned
How transfer learning with a pretrained CNN (MobileNetV2) makes training faster and more accurate on a small dataset
How data augmentation helps a model generalize
How to combine a face detector and a classifier into one real-time pipeline
How model file formats and library versions can break a working project
Limitations
Accuracy depends heavily on lighting, camera angle, and image quality.
The model only knows two classes, so it can't detect incorrectly worn masks.
Faces must be detected first, so heavily occluded or very small faces may be missed.
Acknowledgements
Adrian Rosebrock / PyImageSearch for the original tutorial and project structure
The OpenCV and TensorFlow teams for the libraries
