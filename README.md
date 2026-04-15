# XGenerator
Backend ML inference service with asynchronous task processing (FastAPI + Redis + Celery), designed to handle concurrent workloads and long-running training jobs.
**Scalable ML Inference & Training API Service**  
XGenerator is a scalable backend service for training and serving XGBoost machine learning models via REST APIs. [(Documents)](https://api.xgenerators.net/docs)

## Tech Stack

Backend: FastAPI (Python)  
Queue: Celery + Redis  
Database: Not currently used (planned PostgreSQL integration for persistent storage and scaling)  
Infrastructure: Docker, AWS EC2, Nginx  
ML: XGBoost  

## System Architecture

```
Client (HTTP Requests)
    |
    v
FastAPI API Server
    |
    v
Redis (Message Broker / Task Queue)
    |
    v
Celery Worker (Background Processing)
    |
    v
XGBoost Model (Training / Inference)
```

## Design Highlights

- FastAPI handles HTTP requests and validation
- Celery processes long-running tasks asynchronously
- Redis acts as a message broker between API and workers
- Docker containers isolate services for deployment

### Scalability Design

XGenerator separates API request handling from compute-heavy tasks using Celery workers.

This allows the API server to remain responsive while training jobs are processed asynchronously.

## Key design decisions:

- Task queue architecture for long-running jobs
- Stateless API design for horizontal scaling
- Redis-based message broker for worker coordination
- Containerized services for flexible deployment

## Features

- Dataset Upload - CSV upload with schema inference and validation
- Asynchronous Model Training (Celery + Redis) - XGBoost classification & regression with background task processing
- Batch Prediction API - Predict on multiple rows with missing value handling
- API Key Authentication - User management with tier-based access control
- Rate Limiting - Request throttling to prevent abuse
- Automatic Model Evaluation - 80/20 train/validation split with performance metrics

---

## Quick Start (Docker)

### Prerequisites

- Docker & Docker Compose
- OpenAI API key (optional, for premium tier schema inference)

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/XGenerator.git
cd XGenerator

# Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (optional)
```

### 2. Start Services

```bash
docker compose up -d --build
```

This starts 3 containers:
| Container | Purpose | Port |
|-----------|---------|------|
| `xgenerator_api` | FastAPI server | 8000 |
| `xgenerator_worker` | Celery background tasks | - |
| `xgenerator_redis` | Message queue | 6379 |

### 3. Verify

```bash
# Health check
curl http://localhost:8000/health

# API Documentation
open http://localhost:8000/docs
```

---

## Basic Usage

### 1. Create User

```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"username": "myuser", "email": "user@example.com"}'
```

**Response:**
```json
{
  "user_id": "abc123...",
  "api_key": "sk_live_xxx...",
  "tier": "free"
}
```

> **Save your API key!** It's only shown once.

### 2. Upload Dataset

```bash
curl -X POST http://localhost:8000/datasets \
  -H "X-API-Key: sk_live_xxx..." \
  -F "file=@data.csv" \
  -F "user_id=abc123..." \
  -F "dataset_name=My Dataset"
```

### 3. Train Model

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_live_xxx..." \
  -d '{
    "user_id": "abc123...",
    "model_name": "My Model",
    "dataset_id": "ds_xxx...",
    "task_type": "classification",
    "target": "label"
  }'
```

**Response:** Returns `job_id` (training runs in background)

### 4. Check Training Status

```bash
curl http://localhost:8000/jobs/{job_id}
```

### 5. Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_live_xxx..." \
  -d '{
    "model_id": "model_xxx...",
    "rows": [{"feature1": 10, "feature2": "A"}]
  }'
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/users` | Create user (returns API key) |
| GET | `/users` | List users |
| POST | `/datasets` | Upload CSV dataset |
| PATCH | `/datasets/{id}/schema` | Update schema |
| POST | `/train` | Submit training job |
| GET | `/jobs/{job_id}` | Check job status |
| POST | `/predict` | Batch predictions |
| GET | `/models` | List models |
| GET | `/models/{id}` | Get model details |
| PATCH | `/models/{id}` | Update model name |
| DELETE | `/models/{id}` | Delete model |

**Full API docs:** http://localhost:8000/docs

---

## Environment Variables

Create `.env` file:

```bash
OPENAI_API_KEY=sk-...      # Optional: for premium tier schema inference
OPENAI_MODEL=gpt-3.5-turbo # Optional: default model
REDIS_URL=redis://redis:6379/0
TESTING=false              # Set true to disable rate limits
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| POST /users | 12/hour |
| POST /datasets | 5/min |
| POST /train | 3/min |
| POST /predict | 120/min |

---

## Production Deployment

For EC2/Docker deployment guide, see: [`projectHint_deploy.txt`](projectHint_deploy.txt)

---

## API Documentation

### Static Docs Website

API documentation is available as a static HTML site (ReDoc):
- **Live:** https://AndersonTsaiTW.github.io/xgenerator (after deploying to GitHub Pages)
- **Local (development):** Run `python -m http.server 8010 --directory docs/site`, then open http://127.0.0.1:8010

### Generating Documentation

To regenerate the OpenAPI spec and documentation:

```bash
python scripts/export_openapi.py
```

This script:
1. Imports the FastAPI app from `app/main.py`
2. Extracts the OpenAPI 3.1.0 schema
3. Exports to `docs/site/openapi.json`
4. The `docs/site/index.html` automatically loads and renders it with ReDoc

### CI/CD

API docs are automatically published to GitHub Pages on every push to `main`:
- GitHub Actions workflow: [`.github/workflows/publish-docs.yml`](.github/workflows/publish-docs.yml)
- Trigger: Push to `main` branch or manual workflow dispatch
- Deployment: Static files from `docs/site/` uploaded to GitHub Pages

---

## Project Structure

```
XGenerator/
├── app/
│   ├── main.py              # FastAPI entry
│   ├── config.py            # Configuration
│   ├── celery_app.py        # Celery config
│   ├── routers/             # API endpoints
│   ├── services/            # Business logic
│   ├── tasks/               # Celery tasks
│   ├── models/              # Pydantic schemas
│   └── utils/               # Utilities
├── data/                    # Data storage (gitignored)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── ProjectHint.txt          # Internal dev documentation (Chinese)
```

---

## License

MIT

---

## Documentation

- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **Internal Dev Guide:** [ProjectHint.txt](ProjectHint.txt) (Chinese)
- **Deployment Guide:** [projectHint_deploy.txt](projectHint_deploy.txt) (Chinese)
