# Campus Network Intrusion Detection & Security Monitoring Platform (Backend)

Production-grade FastAPI backend for a cybersecurity monitoring platform with live packet monitoring, threat detection, incident response, and realtime dashboard feeds.

## Tech Stack

- FastAPI + SQLAlchemy
- PostgreSQL
- Scapy (live packet capture)
- Scikit-learn Isolation Forest (anomaly detection)
- WebSockets for realtime security alerts

## Architecture

```text
app/
├── main.py
├── config.py
├── database.py
├── core/
│   ├── security_score.py
│   ├── threat_classifier.py
│   └── anomaly_detector.py
├── monitoring/
│   ├── packet_sniffer.py
│   ├── traffic_analyzer.py
│   └── feature_extractor.py
├── alerts/
│   ├── alert_manager.py
│   ├── websocket_manager.py
│   └── severity_engine.py
├── incidents/
│   ├── incident_service.py
│   └── incident_routes.py
├── analytics/
│   ├── statistics_engine.py
│   └── traffic_metrics.py
├── routes/
│   ├── traffic_routes.py
│   ├── threat_routes.py
│   ├── analytics_routes.py
│   └── dashboard_routes.py
├── models/
│   ├── incident.py
│   ├── traffic_log.py
│   └── threat_event.py
└── utils/
   ├── logger.py
   └── helpers.py
```

## Detection Pipeline

1. Packet is captured (`Scapy` or simulator mode).
2. Packet features are extracted (IPs, protocol, ports, size, temporal behavior).
3. Threat engine runs:

- Rule detection: Port scan, brute-force-like repeated attempts, packet bursts.
- ML detection: Isolation Forest anomaly detection.

1. Threat classification and severity assignment (`LOW|MEDIUM|HIGH|CRITICAL`).
1. Persist:

- Raw packet metadata to `traffic_logs`
- Threat to `threat_events`
- Incident to `incidents`

1. Realtime alert push on WebSocket `/ws/security-alerts`.

## Database Schema (Alembic Managed)

### `incidents`

- `id`
- `timestamp`
- `source_ip`
- `destination_ip`
- `attack_type`
- `severity`
- `description`
- `status`
- `resolved_at`

### `traffic_logs`

- `id`
- `timestamp`
- `source_ip`
- `destination_ip`
- `protocol`
- `source_port`
- `destination_port`
- `packet_size`

### `threat_events`

- `id`
- `timestamp`
- `source_ip`
- `destination_ip`
- `attack_type`
- `severity`
- `threat_score`
- `protocol`
- `destination_port`
- `incident_id`

## API Endpoints

- `GET /api/traffic/live`
- `GET /api/incidents`
- `POST /api/incidents/resolve`
- `GET /api/security-score`
- `GET /api/threats/top`
- `GET /api/analytics/traffic`
- `GET /api/analytics/attacks`

Additional helper APIs:

- `GET /api/dashboard/overview`
- `GET /api/stats`
- `POST /api/alerts/simulate/{attack_type}` (`port-scan`, `brute-force`, `traffic-spike`, `all`)
- `GET /api/alerts/recent`

## WebSocket

- Primary: `ws://localhost:8000/ws/security-alerts`
- Backward-compatible alias: `ws://localhost:8000/ws/alerts`

## Environment Variables

Create `.env` in project root:

```env
APP_NAME=Campus IDS Backend
DEBUG=false
API_PREFIX=/api
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/campus_ids
AUTO_CREATE_SCHEMA=false
CORS_ORIGINS=*
MODEL_PATH=app/ml/iforest_model.joblib
ENABLE_PACKET_SNIFFER=true
SIMULATION_MODE=true
SNIFFER_INTERFACE=
PACKET_BUFFER_SIZE=500
FEATURE_WINDOW_SECONDS=15
PERSIST_JSON_LOGS=false
JSON_STORAGE_DIR=data/runtime
TRAFFIC_JSON_FILE=traffic_logs.jsonl
ALERTS_JSON_FILE=alerts.jsonl
PORT_SCAN_THRESHOLD=8
BRUTE_FORCE_THRESHOLD=10
PACKET_BURST_THRESHOLD=50
```

### Optional JSON File Storage

If you want filesystem-based evidence alongside PostgreSQL, enable:

```env
PERSIST_JSON_LOGS=true
```

Then the backend appends newline-delimited JSON (`.jsonl`) files at:

- `data/runtime/traffic_logs.jsonl`
- `data/runtime/alerts.jsonl`

## Run Locally

Python runtime note:

- Use Python `3.12` (recommended) or `3.11`.
- Python `3.14` may fail to install scientific dependencies (`scikit-learn`/`scipy`) in this stack.

Option A (pyenv, recommended):

```bash
brew install pyenv
pyenv install 3.12.9
pyenv local 3.12.9
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Option B (Homebrew Python 3.12):

```bash
brew install python@3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Create PostgreSQL DB:

```sql
CREATE DATABASE campus_ids;
```

1. Apply database migrations:

```bash
alembic upgrade head
```

1. (Optional) train model:

```bash
python -m app.ml.train_model
```

1. Start API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Migration helpers:

```bash
alembic current
alembic history
alembic revision --autogenerate -m "describe change"
```

## Attack Simulation (Demo)

```bash
curl -X POST http://localhost:8000/api/alerts/simulate/port-scan
curl -X POST http://localhost:8000/api/alerts/simulate/brute-force
curl -X POST http://localhost:8000/api/alerts/simulate/traffic-spike
curl -X POST http://localhost:8000/api/alerts/simulate/all
```

## Notes

- Logging is JSON-based for SIEM/observability pipelines.
- Security score is weighted by open incident severity.
- This setup is simulation-ready and can run with real packet capture when permissions and interface are configured.
- `AUTO_CREATE_SCHEMA=true` can be used for local bootstrap only; keep it `false` in production and use Alembic migrations.
