# Crop Disease Detection

A Streamlit app for predicting crop leaf diseases from uploaded images.

## Features
- Upload a leaf image
- Predict disease class using a trained TensorFlow model
- View confidence score
- Show treatment suggestions for known diseases

## Project structure
- `app.py` — Streamlit web app
- `train_model.py` — training pipeline
- `build_model.py` — alternative model definition
- `predict_image.py` — CLI prediction helper
- `treatments.py` — treatment messaging
- `dataset/` — training/validation image folders
- `label_map.json` — model class mapping
- `crop_disease_model.h5` — trained model artifact

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   . .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train a model:
   ```bash
   python train_model.py
   ```
4. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Docker
Build:
```bash
docker build -t crop-disease-app .
```
Run:
```bash
docker run -p 8501:8501 -v ${PWD}:/app crop-disease-app
```

## Render deployment (free and simple)
1. Push this repo to GitHub.
2. Create a new Render Web Service and connect the repository.
3. Use the default settings or the included `render.yaml`.
4. Render will run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
   ```
5. Set environment variables if your model is stored outside the repo:
   - `MODEL_PATH`
   - `LABEL_MAP_PATH`

## Deployment notes
- The app expects a trained model at `crop_disease_model.h5` and a label map at `label_map.json`.
- In production, mount or copy the model artifact into the container before starting the app.
- For cloud deployment, configure the `MODEL_PATH` and `LABEL_MAP_PATH` environment variables if you store the model outside the project root.
- Render free web services do not support persistent storage, so if the model file is not committed to the repo, store it in a release artifact or another cloud storage location and point `MODEL_PATH` to it.
