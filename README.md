# Titanic Survival Prediction — ML Service

End-to-end ML project demonstrating a production-grade MLOps workflow: data ingestion, model training, experiment tracking, model registry, and serving via a BentoML REST API — containerised and ready for deployment.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Inference (local)](#inference-local)
- [Serving with BentoML](#serving-with-bentoml)
- [Docker](#docker)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## Overview

| Item | Detail |
|---|---|
| **Task** | Binary classification — predict passenger survival on the Titanic |
| **Features** | `pclass`, `sex`, `age`, `fare` |
| **Model** | `SimpleClassifier` — 2-layer feed-forward neural network (PyTorch) |
| **Serving** | BentoML async REST service with adaptive batching |
| **Packaging** | Docker via `bentofile.yaml` |

---

## Architecture

```
Raw Data (Seaborn)
       │
       ▼
 data_loader.py  ──►  StandardScaler  ──►  scaler.joblib
       │
       ▼
   train.py  ──►  titanic_model.pth
       │
       ▼
save_model_bento.py  ──►  BentoML Model Store
                                 │
                                 ▼
                          service.py (TitanicService)
                                 │
                                 ▼
                        POST /predict  (JSON)
```

---

## Project Structure

```
project-01-learn/
├── configs/
│   └── config.yaml          # Hyperparameters & model architecture
├── data/
│   ├── raw/                 # Original, immutable data (git-ignored)
│   ├── processed/           # Cleaned / feature-engineered data
│   └── external/            # Third-party data sources
├── models/
│   ├── model.py             # SimpleClassifier architecture
│   ├── titanic_model.pth    # Trained model weights (git-ignored)
│   └── scaler.joblib        # Fitted StandardScaler (git-ignored)
├── src/
│   ├── data_loader.py       # Data pipeline & preprocessing
│   ├── train.py             # Training loop (supports retrain)
│   ├── inference.py         # Local inference script
│   ├── save_model_bento.py  # Register model/scaler to BentoML store
│   ├── service.py           # BentoML service definition
│   └── api.py               # (Optional) FastAPI wrapper
├── tests/
│   ├── test_env.py          # Environment sanity checks
│   └── generate_dummy_model.py
├── bentofile.yaml           # BentoML build spec
├── Dockerfile               # Container image definition
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.x |
| BentoML | 1.4.x |
| scikit-learn | ≥ 1.3 |

---

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd project-01-learn

# 2. Create and activate virtual environment
python -m venv env_ai
# Windows
env_ai\Scripts\activate
# Linux / macOS
source env_ai/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python src/train.py

# 5. Register model artifacts into BentoML store
python src/save_model_bento.py

# 6. Start the REST API
bentoml serve src.service:TitanicService --reload
```

The API will be available at `http://localhost:3000`.

---

## Configuration

All hyperparameters are managed in [`configs/config.yaml`](configs/config.yaml):

```yaml
model:
  input_size: 4      # Number of input features
  hidden_size: 32    # Hidden layer neurons
  num_classes: 2     # Survived / Deceased

training:
  learning_rate: 0.001
  epochs: 50
  batch_size: 64
```

---

## Training

```bash
python src/train.py
```

- Loads Titanic dataset via Seaborn
- Applies `LabelEncoder` (sex) and `StandardScaler` (all features)
- Saves fitted scaler to `models/scaler.joblib`
- **Supports retrain**: automatically loads existing weights if `models/titanic_model.pth` is found
- Saves updated weights to `models/titanic_model.pth`

---

## Inference (local)

```bash
python src/inference.py
```

Runs a single prediction using a hardcoded sample (`pclass=1, sex=female, age=38, fare=71.28`). Edit the `raw_input` variable inside the script to test different passengers.

---

## Serving with BentoML

### Register artifacts (required once after training)

```bash
python src/save_model_bento.py
```

### Start the server

```bash
bentoml serve src.service:TitanicService --reload
```

### Build a Bento (for production packaging)

```bash
bentoml build
```

### List registered models

```bash
bentoml models list
```

---

## Docker

### Build image via BentoML

```bash
bentoml build
bentoml containerize titanic_service:latest
```

### Run container

```bash
docker run -p 3000:3000 titanic_service:latest
```

---

## API Reference

### `POST /predict`

Predict survival for a single passenger.

**Request body**

```json
{
  "pclass": 1,
  "sex": 0,
  "age": 38.0,
  "fare": 71.28
}
```

| Field | Type | Description |
|---|---|---|
| `pclass` | `int` | Ticket class (1 = First, 2 = Second, 3 = Third) |
| `sex` | `int` | Gender (0 = female, 1 = male) |
| `age` | `float` | Passenger age in years |
| `fare` | `float` | Ticket fare in GBP |

**Response**

```json
{
  "prediction": "Survived"
}
```

`prediction` is either `"Survived"` or `"Deceased"`.

---

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feat/your-feature`
2. Ensure all tests pass: `pytest tests/`
3. Submit a pull request with a clear description of the change
