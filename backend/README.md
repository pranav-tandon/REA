# REA Backend

## Setup
1. Create a virtual environment: `python3 -m venv venv`
2. Activate venv: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Running
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
