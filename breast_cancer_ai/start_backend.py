"""
Start the FastAPI backend server.
Run from the project root:
  python breast_cancer_ai/start_backend.py
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Starting Breast Cancer AI Backend on http://localhost:8000")
print("API docs: http://localhost:8000/docs")
print("Press Ctrl+C to stop\n")

subprocess.run([
    sys.executable, "-m", "uvicorn",
    "breast_cancer_ai.dashboard.main:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload",
], check=True)
