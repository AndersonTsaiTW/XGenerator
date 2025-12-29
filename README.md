- **🔒 Resource Ownership** - Users can only modify their own resources

### Advanced Features
- **Auto ID Exclusion** - Automatically excludes high-cardinality ID columns
- **Date Handling** - Converts date strings to Unix timestamps
- **Missing Value Support** - Predictions work with partial data + warnings
- **Automatic Evaluation** - 80/20 split with comprehensive metrics
- **Pipeline Architecture** - ColumnTransformer + XGBoost for robust preprocessing

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for schema inference)

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd XGenerator

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux/Mac:
# source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the Server

**Windows:**
```powershell
# Navigate to project directory
cd C:\Users\<your-username>\...\XGenerator

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start server
python -m app.main
```

**Linux/Mac:**
```bash
# Navigate to project directory
cd /path/to/XGenerator

# Activate virtual environment
source .venv/bin/activate

# Start server
python -m app.main
```

**Server is ready when you see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc  
- **Health Check**: http://localhost:8000/health

---

## 📖 Complete Workflow Example

### Step 1: Create User Account

First, create a user account to get your API key:

```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "email": "alice@example.com"
  }'
```

**Response:**
```json
{
  "user_id": "user_abc123",
  "username": "alice",
  "email": "alice@example.com",
  "api_key": "sk_live_abc123def456...",
  "created_at": "2024-01-15T10:30:00Z",
  "warning": "⚠️ Save this API key! It won't be shown again."
}
```

**⚠️ IMPORTANT**: Save your API key! It's only shown once upon creation.

### Step 2: Upload Dataset

**Rate Limit**: 5 requests per minute

```bash
curl -X POST http://localhost:8000/datasets \
  -H "X-API-Key: sk_live_abc123..." \
  -F "file=@customer_data.csv" \
  -F "user_id=user_abc123" \
  -F "dataset_name=Customer Churn Q4 2024"
```

**Schema Inference (Tier-Based):**
- **Premium Users** (default): OpenAI GPT-4 analyzes column semantics
  - Intelligently recognizes "age" as numeric, "customer_id" as categorical
  - Considers context, not just data types
- **Free Users**: Pandas dtype analysis with heuristics
  - Still reliable for most use cases

**Response includes:**
- Inferred schema (numeric/categorical features)
- Feature statistics (missing rates, unique counts)
- dataset_id for training

### Step 3: Train Model

**Rate Limit**: 3 requests per minute
**Authentication**: Required

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_live_abc123..." \
  -d '{
    "user_id": "user_abc123",
    "model_name": "Churn Predictor v1",
    "dataset_id": "ds_xyz789",
    "task_type": "classification",
    "target": "churn",
    "xgb_params": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1
    }
  }'
```

**Response includes:**
- model_id
- Training metrics (accuracy, ROC-AUC for classification)
- Features used
- Warnings (if any features were auto-excluded)

### Step 4: Make Predictions

**Rate Limit**: 120 requests per minute
**Authentication**: Required

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_live_abc123..." \
  -d '{
    "model_id": "model_xyz123",
    "rows": [
      {"age": 35, "gender": "M", "tenure_months": 24}
    ]
  }'
```

---

## 🔒 Rate Limiting

Protection against abuse with tiered limits:

### Public Endpoints (IP-based)
- `POST /users`: 12 per hour
- `GET /users`: 120 per minute
- `GET /models`: 120 per minute

### Authenticated Endpoints (API Key-based)
- `POST /datasets`: 5 per minute
- `POST /train`: 3 per minute (resource-intensive)
- `POST /predict`: 120 per minute
- `PATCH /models`: 30 per minute
- `DELETE /models`: 30 per minute

**Testing**: Rate limits are automatically disabled when `TESTING=true`

### Step 5: List Your Models

```bash
curl http://localhost:8000/models?user_id=user_abc123
```

### Step 6: Update Model Name

```bash
curl -X PATCH http://localhost:8000/models/model_aaa111 \
  -H "X-API-Key: sk_live_abc123..." \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Churn Predictor v2 (Improved)"}'
```

---

## 📚 API Documentation

FastAPI automatically generates interactive API documentation:

### Swagger UI (Interactive Testing)
```
http://localhost:8000/docs
```
- Test all endpoints directly in browser
- View request/response schemas
- Execute API calls with sample data

### ReDoc (Clean Documentation)
```
http://localhost:8000/redoc
```
- Beautiful, readable documentation
- Better for sharing with team
- Organized by tags

### OpenAPI JSON Schema
```
http://localhost:8000/openapi.json
```
- Import into Postman/Insomnia
- Generate client SDKs
- CI/CD integration

---

## 🔧 API Endpoints

### Datasets
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/datasets` | Upload CSV dataset with OpenAI schema inference |
| PATCH | `/datasets/{dataset_id}/schema` | Manually update dataset schema |

### Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Train new XGBoost model |
| POST | `/models/{model_id}/retrain` | Retrain existing model with new data |

### Prediction
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Batch predictions (handles missing features) |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/models` | List all trained models |
| GET | `/models/{model_id}` | Get model metadata and metrics |
| GET | `/models/{model_id}/schema` | Get model input schema |

---

## 📝 Example Workflow

### 1. Upload Dataset

```bash
curl -X POST "http://localhost:8000/datasets" \
  -F "file=@data.csv"
```

**Response:**
```json
{
  "dataset_id": "abc123...",
  "filename": "data.csv",
  "row_count": 55638,
  "column_count": 30,
  "schema": {
    "numeric_features": ["age", "price", ...],
    "categorical_features": ["gender", "category", ...]
  }
}
```

### 2. Train Model

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "abc123...",
    "task_type": "classification",
    "target": "label",
    "xgb_params": {
      "n_estimators": 100,
      "max_depth": 5,
      "learning_rate": 0.1
    }
  }'
```

**Response:**
```json
{
  "model_id": "xyz789...",
  "task_type": "classification",
  "features": ["age", "gender", ...],
  "excluded_features": ["customer_id", "product_id"],
  "warnings": ["Auto-excluded ID columns: ['customer_id', 'product_id']"],
  "metrics": {
    "accuracy": 0.9998,
    "roc_auc": 0.9994
  },
  "training_duration": 1.05
}
```

### 3. Make Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "xyz789...",
    "rows": [
      {"age": 32, "gender": 1, "price": 29.99}
    ]
  }'
```

**Response (with missing features):**
```json
{
  "predictions": [
    {"row_index": 0, "prediction": 0.8765}
  ],
  "warnings": [
    "Row 0: Missing features ['category', 'color'] - will be imputed using training data statistics"
  ]
}
```

---

## 📁 Project Structure

```
XGenerator/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration & directories
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   ├── routers/
│   │   ├── datasets.py         # Dataset upload & schema
│   │   ├── training.py         # Model training
│   │   ├── prediction.py       # Predictions
│   │   └── models.py           # Model management
│   ├── services/
│   │   ├── schema_service.py   # OpenAI schema inference
│   │   ├── training_service.py # Pipeline builder
│   │   └── prediction_service.py # Prediction logic
│   └── utils/
│       ├── file_utils.py       # Atomic file operations
│       ├── feature_utils.py    # Feature filtering
│       └── transformers.py     # Date transformers
├── tests/                      # pytest tests
│   ├── unit/                   # Fast unit tests
│   └── integration/            # End-to-end tests
├── data/                       # Data storage (gitignored)
│   ├── datasets/               # Uploaded CSVs
│   ├── artifacts/              # Trained models (.joblib)
│   └── metadata/               # JSON metadata
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development/test dependencies
├── TESTING.md                  # Testing guide
└── .env                        # Environment variables
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
OPENAI_API_KEY=sk-...          # Required for schema inference
OPENAI_MODEL=gpt-3.5-turbo     # Optional, default: gpt-3.5-turbo
```

### XGBoost Parameter Whitelist

For security, only these parameters are allowed:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `n_estimators` | 1-5000 | Number of boosting rounds |
| `learning_rate` | 0.0001-1.0 | Step size shrinkage |
| `max_depth` | 1-16 | Maximum tree depth |
| `subsample` | 0.0-1.0 | Subsample ratio |
| `colsample_bytree` | 0.0-1.0 | Column subsample ratio |

---

## 🎯 Key Features Explained

### Flexible Missing Value Handling

Predictions work even with missing features:

```json
// Training used 27 features
// But you can predict with only 5!
{
  "model_id": "xyz",
  "rows": [{"age": 30, "price": 19.99}]  // Missing 25 features
}

// Response includes warnings
{
  "predictions": [...],
  "warnings": ["Missing features will be imputed using training statistics"]
}
```

### Auto ID Column Exclusion

High-cardinality ID columns are automatically excluded:

```json
{
  "excluded_features": ["customer_id", "product_id"],
  "warnings": ["Auto-excluded ID columns to prevent overfitting"]
}
```

### Date-to-Timestamp Conversion

Date strings are automatically converted to numeric timestamps:

```
"purchase_date": "2024-07-18" → Unix timestamp (numeric)
```

---

## 📊 Model Evaluation

All models are automatically evaluated using:

- **80/20 train/validation split** (stratified for classification)
- **Metrics calculated:**
  - Classification: `accuracy`, `roc_auc` (binary only)
  - Regression: `mae`, `rmse`

---

## 🧪 Testing

See **[TESTING.md](TESTING.md)** for comprehensive testing guide.

**Quick start:**
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run only fast unit tests
pytest tests/unit -m unit

# Run with coverage report
pytest --cov=app --cov-report=html
```

---

## 🚢 Deployment

### Local Development
```bash
python -m app.main
```

### Production (with Uvicorn)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Future)
```bash
docker-compose up --build
```

---

## 🔒 Security Considerations

- **Parameter Validation**: XGBoost params are whitelisted
- **Atomic File Writes**: Prevents data corruption
- **Schema Validation**: Strict request/response validation
- **No Authentication**: ⚠️ Add auth before production deployment

---

## 📈 Performance

- **Training**: < 1 second for 55K rows
- **Predictions**: < 100ms for batch of 100 rows
- **99.98% accuracy** on test dataset

---

## 🛠️ Development

### Code Organization
- **Routers**: HTTP endpoint handlers
- **Services**: Business logic
- **Utils**: Shared utilities
- **Models**: Pydantic schemas

### Adding New Features
1. Define Pydantic schemas in `app/models/schemas.py`
2. Implement service logic in `app/services/`
3. Create router in `app/routers/`
4. Update `app/main.py` to include router
5. Add unit tests in `tests/unit/`

---

## 📄 License

MIT

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Write tests
4. Submit pull request

---

## 📧 Support

For issues or questions, please open a GitHub issue.
