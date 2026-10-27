# Dog–Cat Image Classification Using 3-Layer CNN

**Student Project:** Convolutional Neural Network (CNN) for Dog vs Cat Image Classification  
**Author:** Kalaipriya S  
**Technology Stack:** Python, TensorFlow / Keras, NumPy, Matplotlib, Scikit-learn, VS Code  

---

## 1. Project Objective

The goal of this project is to build and train a Convolutional Neural Network (CNN) using **exactly three convolutional (`Conv2D`) layers** to classify images into two categories:
- **Cat (Class 0)**
- **Dog (Class 1)**

The model takes 128×128 RGB images, extracts spatial visual features across 3 convolutional blocks, and predicts the class probability using a sigmoid output unit.

---

## 2. CNN Architecture

The network has exactly 3 Convolutional layers followed by dense classification layers:

1. **Input Layer:** Shape `(128, 128, 3)` (pixel values normalized to range 0.0 – 1.0).
2. **Convolutional Layer 1:** 32 filters (3×3), ReLU activation, same padding + MaxPooling (2×2).
3. **Convolutional Layer 2:** 64 filters (3×3), ReLU activation, same padding + MaxPooling (2×2).
4. **Convolutional Layer 3:** 128 filters (3×3), ReLU activation, same padding + MaxPooling (2×2).
5. **Flatten Layer:** Reshapes the 3D feature maps into a 1D vector (16 × 16 × 128 = 32,768 features).
6. **Fully Connected Dense Layer:** 128 neurons with ReLU activation.
7. **Dropout Layer:** Rate = 0.5 (to reduce overfitting).
8. **Output Layer:** 1 neuron with Sigmoid activation (outputs probability between 0 and 1).

**Optimizer & Loss:**
- **Optimizer:** Adam (learning rate = 0.001)
- **Loss Function:** Binary Cross-Entropy
- **Metric:** Accuracy

---

## 3. Project Structure

```text
CNN_Dog_Cat_Classification/
│
├── .gitignore                      # Ignores virtual env, cache, and heavy datasets
├── requirements.txt                # Required Python packages
├── README.md                       # Project documentation
│
├── model.py                        # Model architecture & 3-layer validation
├── data_loader.py                  # Dataset downloading, preprocessing & train/val/test splits
├── train.py                        # Model training script with callbacks
├── evaluate.py                     # Model evaluation script (test accuracy, confusion matrix)
├── predict.py                      # Single image inference script
├── main.py                         # Unified command-line runner
│
├── CNN_Dog_Cat_Classification.ipynb# Google Colab notebook for optional cloud GPU training
│
├── data/
│   └── sample_images/              # Sample images for testing (sample_cat.jpg, sample_dog.jpg)
│
├── models/                         # Saved trained model (.keras)
│
└── results/                        # Saved plots, confusion matrix, and evaluation metrics
```

---

## 4. Setup and Execution (VS Code)

### Step 1: Open in VS Code
Open the project directory in VS Code:
```powershell
code C:\Users\HP\Desktop\CNN_Dog_Cat_Classification
```
Open the integrated terminal in VS Code using `Ctrl + ` ` (backtick).

### Step 2: Install Requirements
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Scripts

#### 1. Verify Model Architecture
Checks that the CNN has exactly 3 Conv2D layers:
```powershell
python main.py verify
```

#### 2. Download and Prepare Data
Downloads the Cats vs Dogs dataset and creates train (2,000 images), validation (500 images), and test (500 images) sets:
```powershell
python main.py prepare-data
```

#### 3. Train the Model
Trains the model with EarlyStopping and ModelCheckpoint:
```powershell
python main.py train --epochs 15 --batch-size 32
```
*The trained weights are saved to `models/dog_cat_cnn_model.keras`, and training curves are saved to `results/training_history.png`.*

#### 4. Evaluate the Model
Evaluates the trained model on the unseen test set:
```powershell
python main.py evaluate
```
*This saves the confusion matrix heatmap (`results/confusion_matrix.png`) and the evaluation report (`results/evaluation_metrics.json`).*

#### 5. Predict on an Image
To classify a sample image:
```powershell
python main.py predict --image-path data/sample_images/sample_cat.jpg
```

---

## 5. Google Colab (Optional Cloud Training)

If you prefer to train on Google Colab with a free GPU:
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `CNN_Dog_Cat_Classification.ipynb`.
3. Set the runtime to GPU: **Runtime** > **Change runtime type** > **T4 GPU**.
4. Run all cells (`Ctrl + F9`). The notebook runs independently in the cloud.

---

## 6. Results

Once trained, output files are generated automatically in the `results/` folder:
- `results/training_history.png` – Training vs validation accuracy and loss curves.
- `results/confusion_matrix.png` – Confusion matrix heatmap.
- `results/classification_report.txt` – Precision, recall, and F1-score report.
- `results/evaluation_metrics.json` – Test loss and accuracy summary.

*(Note: Run `python main.py train` to train the model and generate these output files on your machine).*
