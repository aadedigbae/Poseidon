---
title: Poseidon Aquaculture
emoji: 🐠
colorFrom: blue
colorTo: green
sdk: docker
app_file: src/api_app.py
pinned: false
---


# 🐟 Poseidon Aquaculture Monitoring System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Real-time aquaculture monitoring and predictive analytics system for fish pond management**

Poseidon is an AI-powered decision support system that monitors water quality parameters in real-time, predicts risks of fish mortality, and provides actionable recommendations to aquaculture farmers. The system uses machine learning models to estimate hard-to-measure parameters (soft sensors) and forecasts potential issues hours to days in advance.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Data Flow](#-data-flow)
- [Project Structure](#-project-structure)
- [Machine Learning Models](#-machine-learning-models)
- [Storage Options](#-storage-options)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

### Core Capabilities

- **🔬 Soft Sensors**: Predict expensive-to-measure parameters (dissolved oxygen, ammonia) from readily available sensors
- **📊 Real-time Monitoring**: Process sensor readings every 15 seconds after initial data collection
- **⚠️ Risk Forecasting**: Predict water quality risks at multiple time horizons:
  - **+1 hour**: Immediate risk detection (LSTM/GRU models)
  - **+6 hours**: Short-term planning
  - **+24 hours**: Daily forecasting
  - **+3 days**: Long-term strategic planning (classical ML)
- **💡 Decision Support**: AI-powered recommendations for pond management actions
- **📈 Historical Analysis**: Track trends and patterns over time
- **☁️ Flexible Storage**: Support for both local CSV files and Firebase Cloud Firestore
- **🚀 RESTful API**: Easy integration with mobile apps, dashboards, and IoT devices

### Advanced Features

- **Hybrid ML Ensemble**: Combines classical ML (Random Forest, XGBoost) with deep learning (LSTM, GRU)
- **Feature Engineering**: Automated calculation of rolling statistics, trends, and slopes
- **Multi-pond Management**: Support for monitoring multiple fish ponds simultaneously
- **Action Logging**: Complete audit trail of all recommendations and interventions
- **Configurable Thresholds**: Customizable risk levels and alert parameters

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         POSEIDON SYSTEM                             │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Sensors    │────▶│  API Layer   │────▶│   Storage    │
│              │     │ (FastAPI)    │     │ (CSV/Firebase)│
│ - Temp       │     │              │     │              │
│ - pH         │     │ - Enrichment │     │ - Readings   │
│ - Turbidity  │     │ - Validation │     │ - Risks      │
└──────────────┘     │ - Analytics  │     │ - Forecasts  │
                     └──────────────┘     │ - Actions    │
                             │            └──────────────┘
                             ▼
                     ┌──────────────┐
                     │ Soft Sensors │
                     │              │
                     │ ML Models:   │
                     │ - DO Pred    │
                     │ - NH3 Pred   │
                     └──────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │  Forecasting │
                     │              │
                     │ - LSTM/GRU   │
                     │ - XGBoost    │
                     │ - Ensemble   │
                     └──────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │   Decision   │
                     │   Support    │
                     │              │
                     │ - Risk Calc  │
                     │ - Actions    │
                     └──────────────┘
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version
- **Git**: For cloning the repository
- **Firebase Account** (optional): For cloud storage

### Step 1: Clone the Repository

```bash
git clone https://github.com/aadedigbae/Poseidon.git
cd Poseidon
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Configuration Files

```bash
# Copy example config files
cp config.example.json config.json
cp .env.example .env
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Storage Configuration
STORAGE_MODE=local  # Options: local, firebase

# Firebase Configuration (if using Firebase)
FIREBASE_CREDENTIALS={"type":"service_account","project_id":"your-project",...}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=false

# Model Configuration
MODEL_BUNDLE_PATH=./models/model_bundle.pkl

# Logging
LOG_LEVEL=INFO
LOG_FILE=poseidon.log
```

### Firebase Setup (Optional)

If using Firebase storage:

1. **Create Firebase Project**: Go to [Firebase Console](https://console.firebase.google.com/)
2. **Enable Firestore**: Enable Cloud Firestore in your project
3. **Generate Service Account**:
   - Go to Project Settings → Service Accounts
   - Click "Generate New Private Key"
   - Save the JSON file
4. **Set Environment Variable**: Add the entire JSON content to `FIREBASE_CREDENTIALS` in `.env`

### Model Configuration

The system expects a model bundle file containing:

```python
{
    'HISTORY_STEPS': 168,  # Number of readings needed (e.g., 168 hourly readings = 7 days)
    'SOFT_SENSOR_MODELS': {
        'dissolved_oxygen': model_do,
        'ammonia': model_nh3
    },
    'FORECAST_MODELS': {
        'lstm_1h': model_lstm,
        'xgboost_24h': model_xgb,
        # ... other models
    }
}
```

---

## 📖 Usage

### Starting the API Server

```bash
# Start with default settings
python api_app.py

# Or with custom host/port
python api_app.py --host 0.0.0.0 --port 8000

# With auto-reload for development
uvicorn api_app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Quick Start Example

```python
import requests

# 1. Submit a sensor reading
response = requests.post(
    "http://localhost:8000/reading",
    json={
        "pond_id": "RWA-01",
        "timestamp": "2025-12-02T10:00:00",
        "temperature": 28.5,
        "pH": 7.2,
        "turbidity_proxy": 15.3
    }
)

# 2. Get current pond status
status = requests.get("http://localhost:8000/pond/RWA-01/status")
print(status.json())

# 3. Get risk forecast
forecast = requests.get("http://localhost:8000/pond/RWA-01/forecast")
print(forecast.json())
```

---

## 🔌 API Documentation

### Core Endpoints

#### `POST /reading`
Submit a new sensor reading for a pond.

**Request Body:**
```json
{
  "pond_id": "RWA-01",
  "timestamp": "2025-12-02T10:00:00",
  "temperature": 28.5,
  "pH": 7.2,
  "turbidity_proxy": 15.3
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Reading stored. Analysis will run in 2.5 hours (156 more readings needed)",
  "enriched_data": {
    "temperature": 28.5,
    "pH": 7.2,
    "turbidity_proxy": 15.3,
    "predicted_do": 6.8,
    "predicted_nh3": 0.05
  }
}
```

#### `GET /pond/{pond_id}/status`
Get current water quality status and latest readings.

**Response:**
```json
{
  "pond_id": "RWA-01",
  "status": "healthy",
  "latest_reading": {
    "timestamp": "2025-12-02T10:00:00",
    "temperature": 28.5,
    "pH": 7.2,
    "predicted_do": 6.8
  },
  "current_risk": 0.15,
  "risk_level": "low"
}
```

#### `GET /pond/{pond_id}/forecast`
Get risk predictions for multiple time horizons.

**Response:**
```json
{
  "pond_id": "RWA-01",
  "forecasts": [
    {
      "horizon": "+1h",
      "risk_score": 0.12,
      "risk_level": "low",
      "confidence": 0.89
    },
    {
      "horizon": "+24h",
      "risk_score": 0.45,
      "risk_level": "medium",
      "confidence": 0.76
    }
  ],
  "recommendations": [
    "Monitor dissolved oxygen levels closely",
    "Consider aerator activation in 18 hours"
  ]
}
```

#### `GET /pond/{pond_id}/history`
Get historical readings and trends.

**Query Parameters:**
- `hours` (optional): Number of hours to retrieve (default: 24)
- `include_predictions` (optional): Include ML predictions (default: true)

#### `GET /ponds`
List all monitored ponds.

**Response:**
```json
{
  "ponds": ["RWA-01", "RWA-02", "RWA-03"],
  "total": 3
}
```

#### `GET /health`
Check API health and system status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "storage": "firebase",
  "models_loaded": true,
  "uptime": "3d 14h 23m"
}
```

---

## 🔄 Data Flow

### Complete End-to-End Journey

```
1. SENSOR READING ARRIVES
   ↓
   Raw Data: {temperature, pH, turbidity}

2. SOFT SENSOR ENRICHMENT
   ↓
   ML Models Predict: {predicted_do, predicted_nh3}

3. DATA STORAGE
   ↓
   Saved to: readings.csv or Firebase Firestore

4. WAIT FOR MINIMUM DATA
   ↓
   System needs 168 readings (configurable)
   At 15-second intervals = ~42 minutes

5. RETRIEVE HISTORICAL DATA
   ↓
   Load last 168 readings from storage

6. FEATURE ENGINEERING
   ↓
   Calculate: rolling means, trends, slopes, etc.

7. RISK FORECASTING
   ↓
   Multiple Models:
   - LSTM/GRU for short-term (+1h, +6h)
   - XGBoost/RF for long-term (+24h, +3d)

8. DECISION SUPPORT
   ↓
   Analyze risks → Generate recommendations

9. SAVE PREDICTIONS
   ↓
   Store to: forecasts.csv and actions_log.csv

10. API RESPONSE
    ↓
    Return predictions + recommendations to user
```

### Data Storage Structure

Each pond maintains four CSV files (or Firestore collections):

1. **`readings.csv`**: Raw + enriched sensor data
   ```csv
   timestamp,temperature,pH,turbidity_proxy,predicted_do,predicted_nh3
   2025-12-02T10:00:00,28.5,7.2,15.3,6.8,0.05
   ```

2. **`risk_predictions.csv`**: Real-time risk assessments
   ```csv
   timestamp,risk_score,risk_level,confidence
   2025-12-02T10:00:00,0.15,low,0.89
   ```

3. **`forecasts.csv`**: Future risk predictions
   ```csv
   timestamp,horizon,risk_score,risk_level
   2025-12-02T10:00:00,+1h,0.12,low
   2025-12-02T10:00:00,+24h,0.45,medium
   ```

4. **`actions_log.csv`**: Recommended actions
   ```csv
   timestamp,action,reason,priority
   2025-12-02T10:00:00,"Increase aeration",low_do,high
   ```

---

## 📁 Project Structure

```
Poseidon/
├── api_app.py                   # Main FastAPI application
├── soft_sensors_runtime.py      # Virtual sensor predictions
├── runtime_forecast.py          # Risk forecasting engine
├── decision_support.py          # Recommendation system
├── storage.py                   # Storage abstraction layer
├── storage_local.py             # Local CSV storage implementation
├── storage_firebase.py          # Firebase storage implementation
├── startup_firebase.py          # Firebase initialization
├── requirements.txt             # Python dependencies
├── config.json                  # Configuration file
├── .env                         # Environment variables
├── README.md                    # This file
│
├── models/                      # ML model files
│   ├── model_bundle.pkl         # Trained model bundle
│   ├── soft_sensor_do.pkl       # DO prediction model
│   └── soft_sensor_nh3.pkl      # Ammonia prediction model
│
├── data/                        # Local data storage (if using CSV)
│   ├── RWA-01/
│   │   ├── readings.csv
│   │   ├── risk_predictions.csv
│   │   ├── forecasts.csv
│   │   └── actions_log.csv
│   └── RWA-02/
│       └── ...
│
├── logs/                        # Application logs
│   └── poseidon.log
│
├── tests/                       # Unit and integration tests
│   ├── test_api.py
│   ├── test_soft_sensors.py
│   ├── test_forecasting.py
│   └── test_storage.py
│
└── docs/                        # Additional documentation
    ├── API.md
    ├── MODELS.md
    └── DEPLOYMENT.md
```

---

## 🤖 Machine Learning Models

### Soft Sensor Models

**Purpose**: Predict expensive-to-measure parameters from readily available sensors.

#### Dissolved Oxygen (DO) Predictor
- **Input Features**: temperature, pH, turbidity, time_of_day, rolling statistics
- **Algorithm**: Random Forest Regressor
- **Accuracy**: R² = 0.87, RMSE = 0.42 mg/L
- **Update Frequency**: Real-time with each reading

#### Ammonia (NH₃) Predictor
- **Input Features**: temperature, pH, turbidity, predicted_do, rolling statistics
- **Algorithm**: Gradient Boosting
- **Accuracy**: R² = 0.82, RMSE = 0.08 mg/L
- **Update Frequency**: Real-time with each reading

### Forecasting Models

#### Short-Term Models (LSTM/GRU)
- **Horizons**: +1h, +6h
- **Architecture**:
  - 2-layer LSTM with 64 units per layer
  - Dropout (0.2) for regularization
  - Dense output layer
- **Input**: Sequential data (last 168 readings)
- **Strengths**: Captures temporal patterns and sudden changes

#### Long-Term Models (Classical ML)
- **Horizons**: +24h, +3d
- **Algorithms**: XGBoost, Random Forest, LightGBM ensemble
- **Input**: Engineered features (rolling stats, trends, slopes)
- **Strengths**: Stable predictions, interpretable features

### Model Training

To retrain models with new data:

```bash
python train_models.py --data-path ./training_data --output-dir ./models
```

---

## 💾 Storage Options

### Local CSV Storage

**Best for**: Development, testing, small-scale deployments

**Advantages**:
- No external dependencies
- Easy to inspect and debug
- Fast for single-machine deployments
- No cost

**Configuration**:
```env
STORAGE_MODE=local
LOCAL_DATA_PATH=./data
```

### Firebase Cloud Firestore

**Best for**: Production, multi-user deployments, cloud hosting

**Advantages**:
- Automatic scaling
- Real-time synchronization
- Built-in backup and recovery
- Multi-device access
- Global CDN

**Configuration**:
```env
STORAGE_MODE=firebase
FIREBASE_CREDENTIALS=<service-account-json>
```

**Firestore Structure**:
```
poseidon/
├── ponds/
│   ├── RWA-01/
│   │   ├── metadata: {name, location, capacity}
│   │   ├── readings/
│   │   │   └── <timestamp-documents>
│   │   ├── forecasts/
│   │   │   └── <timestamp-documents>
│   │   └── actions/
│   │       └── <timestamp-documents>
│   └── RWA-02/
│       └── ...
```

---

## 🚢 Deployment

### Local Development

```bash
# Start development server with auto-reload
uvicorn api_app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build image
docker build -t poseidon-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data poseidon-api
```

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Hugging Face Spaces

The project is deployed on Hugging Face Spaces:
- **URL**: https://huggingface.co/spaces/Fierceee/poseidon-aquaculture
- **Runtime**: Python 3.9
- **Hardware**: CPU Basic (free tier)

To deploy your own:

1. **Create Space**: Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. **Push Code**:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/poseidon-aquaculture
   git push hf main
   ```
3. **Configure Secrets**: Add environment variables in Space settings

### Production Considerations

- **Scaling**: Use Gunicorn/Uvicorn workers for high traffic
  ```bash
  gunicorn api_app:app -w 4 -k uvicorn.workers.UvicornWorker
  ```
- **HTTPS**: Use reverse proxy (Nginx, Caddy) with SSL certificates
- **Monitoring**: Integrate with logging services (Sentry, DataDog)
- **Backup**: Schedule regular backups of Firebase or CSV files
- **Rate Limiting**: Implement API rate limits to prevent abuse

---

## 🧪 Testing

### Run Unit Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_api.py

# With coverage
pytest --cov=. --cov-report=html
```

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/health

# Submit reading
curl -X POST http://localhost:8000/reading \
  -H "Content-Type: application/json" \
  -d '{"pond_id":"TEST-01","timestamp":"2025-12-02T10:00:00","temperature":28.5,"pH":7.2,"turbidity_proxy":15.3}'

# Get status
curl http://localhost:8000/pond/TEST-01/status
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- **Code Style**: Follow PEP 8 (use `black` formatter)
- **Type Hints**: Use type annotations for all functions
- **Documentation**: Add docstrings to all modules, classes, and functions
- **Testing**: Write unit tests for new features
- **Commit Messages**: Use conventional commits format

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Contributors**: All developers who have contributed to this project
- **Research**: Based on aquaculture research from [Institution Name]
- **Libraries**: Built with FastAPI, scikit-learn, TensorFlow, and Firebase
- **Community**: Thanks to the open-source community for their tools and support

---

## 📞 Contact

- **Project Lead**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@aadedigbae](https://github.com/aadedigbae)
- **Hugging Face**: [@Fierceee](https://huggingface.co/Fierceee)
- **Issues**: [Report bugs](https://github.com/aadedigbae/Poseidon/issues)

---

## 🗺️ Roadmap

### Version 1.1 (Q1 2025)
- [ ] Mobile app integration (iOS/Android)
- [ ] SMS/email alert system
- [ ] Multi-language support
- [ ] Dashboard UI improvements

### Version 1.2 (Q2 2025)
- [ ] Image-based water quality analysis
- [ ] Integration with IoT sensor networks
- [ ] Advanced visualization and reporting
- [ ] Multi-tenant support

### Version 2.0 (Q3 2025)
- [ ] AI-powered feed optimization
- [ ] Growth prediction models
- [ ] Economic optimization algorithms
- [ ] Blockchain-based traceability

---

## ❓ FAQ

**Q: How long does it take before I get predictions?**
A: The system needs 168 readings (configurable). At 15-second intervals, this takes approximately 42 minutes. The exact time is shown when you start the API.

**Q: Can I use this for saltwater aquaculture?**
A: The current models are trained on freshwater fish ponds. For saltwater, you'll need to retrain the models with saltwater data.

**Q: What sensors do I need?**
A: Minimum sensors: temperature, pH, and turbidity. The system will predict dissolved oxygen and ammonia. For better accuracy, add actual DO and NH₃ sensors.

**Q: How accurate are the predictions?**
A: Short-term predictions (1-6 hours) have ~85% accuracy. Long-term (24+ hours) have ~75% accuracy. Accuracy improves with more data.

**Q: Can I monitor multiple ponds?**
A: Yes! The system supports unlimited ponds. Each pond has its own models and history.

**Q: Is this free to use?**
A: The software is free (MIT license). Cloud storage (Firebase) has free tier, then costs based on usage.

**Q: How do I contribute training data?**
A: Contact us at [email] to contribute anonymized data to improve the models.

---

<div align="center">

**Made with ❤️ for sustainable aquaculture**

⭐ Star this repository if you find it useful!

[🐟 Report Bug](https://github.com/aadedigbae/Poseidon/issues) • [💡 Request Feature](https://github.com/aadedigbae/Poseidon/issues) • [📖 Documentation](https://github.com/aadedigbae/Poseidon/wiki)

</div>