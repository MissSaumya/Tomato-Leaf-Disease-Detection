# Tomato Plant Leaf Disease Classification using EfficientNet-B0

## Dataset
The dataset consists of images of tomato leaves categorized into different disease types. The dataset is taken from a newly augmented version of the Pant Village dataset. From this dataset, only the tomato portion has been extracted, as tomatoes are an important vegetable. 

The dataset includes 10 classifications, structured into 10 folders:
- 9 folders contain images of diseased tomato leaves.
- 1 folder contains images of healthy tomato leaves.

The dataset is organized into:
- **Training Set:** Contains images of diseased and healthy tomato leaves, categorized into 10 separate folders.
- **Validation Set:** Contains images following the same classification structure as the training set for model evaluation.
- **Testing Set:** Contains images of tomato leaves labeled with their disease name directly in the image filename, rather than being stored in separate folders.

## Abstract
Tomato plant diseases significantly impact crop yield and quality. Detecting and classifying these diseases at an early stage can help in better crop management and prevent large-scale losses.

Deep learning models, particularly Convolutional Neural Networks (CNNs), provide effective solutions for automated disease detection. By leveraging CNN architectures, we can extract relevant features from leaf images and classify diseases with high accuracy.

In this project, EfficientNet-B0 is used due to its high accuracy and computational efficiency. EfficientNet-B0 uses a compound scaling technique to balance model depth, width, and resolution, making it a suitable choice for mobile and embedded applications.

The model is trained using a labeled dataset containing diseased and healthy tomato leaf images. Performance evaluation is done using standard metrics such as accuracy, precision, recall, and F1-score.

The final system can be integrated into a web or mobile application for real-time disease detection, aiding farmers and agricultural professionals in identifying plant diseases quickly and efficiently.

## Project Objectives
- **Develop an AI-based system** to classify tomato plant leaf diseases using EfficientNet-B0.
- **Utilize an augmented dataset** derived from the Pant Village dataset, specifically focusing on tomato leaves.
- **Train and evaluate the model** using multiple performance metrics.
- **Deploy the model** into a real-world application for practical usage in agriculture.

## Folder Structure
```
├── Dataset
│   ├── train
│   │   ├── Tomato_Healthy
│   │   ├── Tomato_Bacterial_Spot
│   │   ├── Tomato_Early_Blight
│   │   ├── Tomato_Late_Blight
│   │   ├── Tomato_Leaf_Mold
│   │   ├── Tomato_Septoria_Leaf_Spot
│   │   ├── Tomato_Spider_Mites
│   │   ├── Tomato_Target_Spot
│   │   ├── Tomato_Tomato_Mosaic_Virus
│   │   ├── Tomato_Tomato_Yellow_Leaf_Curl_Virus
│   ├── validation
│   │   ├── (same structure as train folder)
│   ├── test
│   │   ├── Individual image files labeled with the disease name in their filename
```

## Dataset Download
Due to the large size, the dataset is hosted externally. You can download it from:

🔗 [Kaggle]((https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset))

After downloading, extract the dataset from Tomato folder.

## Usage Instructions
1. **Clone the repository:**
   ```bash
   git clone <repository-link>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model:**
   ```bash
   python train.py --dataset path/to/dataset
   ```
4. **Evaluate the model:**
   ```bash
   python evaluate.py --model path/to/model
   ```
5. **Deploy the model:**
   - Integrate into a web or mobile application for real-time classification.

## Performance Metrics
The model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

## Future Work
- Improve dataset augmentation techniques.
- Deploy the model as a lightweight API for mobile and edge devices.
- Expand the dataset with more real-world samples for better generalization.




