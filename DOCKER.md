# 🐳 Docker Deployment Guide — Breast Cancer AI Platform

This guide details how to build, run, and manage the full multi-omics breast cancer AI platform using Docker and Docker Compose.

---

## 🏗️ Architecture Overview

The containerized stack consists of two coordinated services connected via a dedicated bridge network:

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Host                            │
│                                                             │
│   ┌───────────────────────────┐   ┌─────────────────────┐   │
│   │   Next.js 16 Frontend     │   │   FastAPI Backend   │   │
│   │   (Node 20 Alpine)        │   │   (Python 3.11)     │   │
│   │   Port: 3000              │   │   Port: 8000        │   │
│   │   Standalone Production   │   │   PyTorch CPU + ML  │   │
│   └─────────────┬─────────────┘   └──────────┬──────────┘   │
│                 │                            │              │
│                 └───────────┬────────────────┘              │
│                             │                               │
│              ┌──────────────┴──────────────┐                │
│              │     Persistent Volumes      │                │
│              │  • breast_cancer_ai.db      │                │
│              │  • experiments/             │                │
│              │  • Clinical Datasets (ro)   │                │
│              └─────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

1. **Backend (`breast-cancer-ai-backend`)**:
   - **Base**: `python:3.11-slim`
   - **Libraries**: FastAPI, Uvicorn, PyTorch (CPU-optimized wheels), PyTorch Geometric (PyG), XGBoost, Scikit-learn, SciPy, NetworkX, SHAP.
   - **Port**: `8000` (`http://localhost:8000`)
   - **Healthcheck**: Regular polling of `GET /health` ensures dependent services start only when the ML engine is fully ready.

2. **Frontend (`breast-cancer-ai-frontend`)**:
   - **Base**: `node:20-alpine` (multi-stage standalone build)
   - **Libraries**: Next.js 16 (Turbopack, App Router), React 19, Tailwind CSS v4, Recharts, Phosphor Icons.
   - **Port**: `3000` (`http://localhost:3000`)
   - **Size**: ~150 MB (thanks to Next.js standalone output tracing).

3. **Data Persistence**:
   - `breast_cancer_ai.db` (SQLite): Stores all 10 pipeline execution steps, historical runs, and clinical prediction records.
   - `experiments/`: Stores trained models (`xgboost_model.joblib`, GAT embeddings, evaluation ROC/PR curves, and JSON metrics).
   - Datasets: Real clinical multi-omics files (`ISPY2`, `GSE122630`, `GSE240671`) are mounted read-only into `/app/`.

---

## 🚀 Quick Start (Single Command)

### 1. Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS) or Docker Engine + Docker Compose (Linux).
- Ensure Docker Desktop is started and running.

### 2. Launch the Platform
From the repository root directory, execute:

```bash
docker compose up --build
```

To run in detached (background) mode:

```bash
docker compose up -d --build
```

### 3. Access the Applications
- 🖥️ **Web Dashboard**: Open [http://localhost:3000](http://localhost:3000)
- 📖 **Interactive API Docs (Swagger UI)**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- 🩺 **Backend Healthcheck**: Open [http://localhost:8000/health](http://localhost:8000/health)

---

## 🛠️ Management Commands

### Viewing Real-Time Logs
```bash
# Follow all container logs
docker compose logs -f

# Follow only backend logs
docker compose logs -f backend

# Follow only frontend logs
docker compose logs -f frontend
```

### Checking Container Health & Status
```bash
docker compose ps
```

### Stopping the Platform
```bash
# Stop containers while preserving database and models
docker compose down

# Stop containers and remove networks
docker compose down -v
```

### Restarting a Specific Service
```bash
docker compose restart backend
docker compose restart frontend
```

---

## 🔒 Configuration & Environment Variables

Key parameters can be customized directly in `docker-compose.yml` or via a `.env` file:

| Variable | Service | Default | Description |
|---|---|---|---|
| `PORT` | Backend / Frontend | `8000` / `3000` | Port listening inside the container |
| `CORS_ORIGINS` | Backend | `http://localhost:3000,http://127.0.0.1:3000` | Allowed origins for cross-origin API calls |
| `DATABASE_URL` | Backend | `sqlite:///./breast_cancer_ai.db` | SQLAlchemy database connection URI |
| `NEXT_PUBLIC_API_URL` | Frontend | `http://localhost:8000` | Target backend URL for browser requests |

---

## 💡 Notes on Lightweight Builds

- **No Multi-Gigabyte CUDA Bloat**: PyTorch is installed using the official CPU wheel index (`--index-url https://download.pytorch.org/whl/cpu`). This reduces backend image size from ~7 GB down to ~1.2 GB while delivering instant CPU inference (<50 ms per patient).
- **Next.js Standalone**: Next.js automatically bundles only the exact node_modules dependencies needed for execution, reducing frontend image size from >1 GB down to ~150 MB.
