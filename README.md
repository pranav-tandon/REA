# REA Project

## Project Structure

```
REA
├── backend
│   ├── scrapers
│   │   ├── __init__.py
│   │   ├── zillow_scraper.py
│   │   ├── redfin_scraper.py
│   │   └── ... (other site scrapers)
│   ├── app.py
│   ├── cma.py
│   ├── neighborhood.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── README.md
├── nextjs
│   ├── package.json
│   ├── pages
│   │   ├── api
│   │   │   └── chat.ts
│   │   └── index.tsx
│   ├── public
│   ├── tsconfig.json
│   ├── next.config.js
│   ├── Dockerfile
│   └── README.md
├── docker-compose.yml
├── .gitignore
├── .env.example
└── README.md
```

## Running the Project

### Using Docker (Recommended)

Ensure you have **Docker** and **Docker Compose** installed.

```bash
# Build and start all services (backend, frontend, MongoDB)
docker-compose up --build
```

- Backend (FastAPI) runs on `http://localhost:8005`
- Frontend (Next.js) runs on `http://localhost:3005`
- MongoDB runs on port `27017`

To stop the services:

```bash
docker-compose down
```

---

### Running Without Docker (Bare Metal)

#### Running the Backend

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment and install dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8005
```

#### Running the Frontend

```bash
# Navigate to the frontend directory
cd nextjs

# Install dependencies
npm install

# Start the Next.js development server
npm run dev -- -p 3005
```

## Setting Up a Fresh Environment

If the backend setup does not work, follow these steps:

### 1. Install `pyenv` (Python Version Manager)

```bash
brew install pyenv
```

### 2. Install Python 3.11

```bash
pyenv install 3.11.6
```

### 3. Set Python Version for the Project

```bash
cd /Users/pranav/projects/REA
pyenv local 3.11.6
```

### 4. Create & Activate Virtual Environment

```bash
cd backend
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `torch` installation fails, install it separately:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 6. Run the Backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8005
```

## Ignore Virtual Environment in Git

Add the following to `.gitignore`:

```
venv/
.venv/
```

If `venv/` was previously committed:

```bash
git rm -r --cached venv
git commit -m "Remove venv from tracking and update .gitignore"
```

## Handling Large Files in Git

If you encounter GitHub push errors due to large files (>100MB), follow these steps:

### 1. Install BFG Repo-Cleaner

```bash
brew install bfg
```

### 2. Remove Large Files from Git History

```bash
bfg --delete-files "libtorch_cpu.dylib" .
# OR remove an entire folder
bfg --delete-folders venv .
```

### 3. Clean and Force Push

```bash
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
```

## Pushing Clean Repo

After removing large files:

```bash
git add .
git commit -m "Clean repo and remove large files"
git push origin main
```