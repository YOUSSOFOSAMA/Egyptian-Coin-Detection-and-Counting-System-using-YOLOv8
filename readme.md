\# Egyptian Coin Detection and Counting System using YOLOv8



An end-to-end computer vision application for detecting, classifying, and counting Egyptian coins using \*\*YOLOv8\*\*. The project includes a trained custom object detection model and an interactive \*\*Streamlit\*\* web application that supports image uploads and live camera capture for real-time inference.



\---



\## Features



\- Detects Egyptian coins using a custom-trained YOLOv8 model

\- Supports both image upload and live camera capture

\- Automatic coin classification

\- Automatic coin cropping for each detected coin

\- Centroid-based tracking to prevent duplicate counting

\- Session-based counting statistics

\- Automatic total currency value calculation

\- Interactive Streamlit web interface

\- Fast real-time inference



\---



\## Model Performance



The model was trained using \*\*YOLOv8 Nano (YOLOv8n)\*\* with transfer learning.



| Metric | Score |

|---------|------:|

| mAP@50 | \*\*0.995\*\* |

| mAP@50-95 | \*\*0.824\*\* |

| Precision | \*\*0.993\*\* |

| Recall | \*\*1.000\*\* |



Training Configuration



\- Model: YOLOv8n

\- Epochs: 30

\- Image Size: 640 × 640

\- Hardware: NVIDIA T4 GPU

\- Training Platform: Google Colab



\---



\## Technologies



\- Python

\- YOLOv8

\- Ultralytics

\- Streamlit

\- OpenCV

\- Pillow

\- PyTorch



\---



\## Project Structure



egyptian-coin-detector-yolov8/

│

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── best.pt

│

├── training.ipynb

│

└── Gneh\_Model\_Evaluation\_Report.pdf



\---



\## Installation



Clone the repository



```bash

git clone https://github.com/YOUSSOFOSAMA/egyptian-coin-detector-yolov8.git

```



Navigate to the project directory



```bash

cd egyptian-coin-detector-yolov8

```



Install dependencies



```bash

pip install -r requirements.txt

```



\---



\## Usage



Run the Streamlit application



```bash

streamlit run app.py

```



Then open the displayed local URL in your browser.



\---



\## How It Works



1\. Upload an image or capture one using your camera.

2\. The YOLOv8 model detects and classifies each coin.

3\. Bounding boxes are drawn around detected coins.

4\. Each detected coin is cropped automatically.

5\. A centroid-based tracking algorithm prevents duplicate counting.

6\. Session statistics are updated.

7\. The total monetary value is calculated automatically.



\---



\## Dataset



The model was trained using a custom annotated Egyptian coin dataset.



Classes



\- Gneh

\- NosGneh



Annotation Platform



\- Roboflow



\---



\## Future Improvements



\- Support additional Egyptian pounds

\- Real-time webcam video detection

\- Object tracking across video frames

\- Export detection reports

\- Mobile deployment



\---

