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
   - `MODEL_URL` — public URL used to download the trained model when it is not already present
   - `MODEL_PATH`
   - `LABEL_MAP_PATH`

## Deployment notes
- The app expects a trained model and `label_map.json`. If the model file is missing and `MODEL_URL` is set, it downloads the model once to `MODEL_PATH` before loading it.
- The trained model is hosted as a downloadable asset on the [GitHub Releases page](https://github.com/VijayUchiha2004/Crop_Disease_Detector/releases).
- `MODEL_URL` defaults to the `v1.0.0` release asset URL in `render.yaml` and `.env.example`. Keep that release asset filename as `crop_disease_model.h5`, or update the URL in both files.
- `MODEL_PATH` and `LABEL_MAP_PATH` can be configured when the model and labels are provided through mounted storage instead.
- Render free web services do not provide persistent storage. The model is therefore downloaded to the instance at startup and may be downloaded again after an instance restart.
