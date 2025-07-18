
# Fashion Product Images Prediction

## Overview
This repository contains code for a deep learning model that predicts fashion product attributes (such as color, type, season, and gender) from product images. The model is trained on the Fashion Product Images dataset and includes scripts for training, inference, and a Streamlit-based web UI for image prediction.

---

## Key Files

- **`api/main.py`**  
  Runs the FastAPI server for prediction. Loads the trained model and provides the `/predict` endpoint to infer attributes from an image.

- **`ui/app.py`**  
  Streamlit app for uploading images and displaying predictions by calling the FastAPI backend.

- **`model-training.ipynb`**  
  Contains the model architecture definition (e.g., multi-output CNN with ResNet backbone) and the training.

- **`sample output`**
      Contains the sample output screeenshots of the web ui
- **`label_encoders.pkl`**
     It is necessary for the decoding of model output into the labels
  


---

## Downloading and loading the model
- Download the model from https://drive.google.com/file/d/1BOtH3abvEUHz4scR35o2EHO-2xsbq88C/view?usp=sharing
- paste the path of model in line 71 in api/main.py

---

## Running the Project

1. **Install requirements:**  
   ```bash
   pip install -r requirements.txt

2.**Run the API server:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
3.**Run the Streamlit UI:**
In a new terminal:

```bash
streamlit run ui/app.py
```
Open the Streamlit app in your browser (usually at http://localhost:8501). Upload an image, and the model will predict fashion attributes by querying the API.















