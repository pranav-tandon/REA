```markdown
# REA Project

```
README.md
```README.md
# REA Project
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
│   └── README.md
├── .gitignore
├── .env.example
└── README.md
```

## Running the Project

### To Run the Backend

```bash
# Navigate to the Next.js directory
cd /Users/pranav/projects/REA/nextjs

# Install dependencies
npm install

# Start the development server
npm run dev
```

### To Run the Frontend

```bash
# Navigate to the Next.js directory
cd /Users/pranav/projects/REA/nextjs

# Install dependencies
npm install

# Start the development server
npm run dev
```

## Creating a New Environment

If the above steps for running the backend do not work, follow these steps to create a fresh environment.

### 1. Install pyenv (Manage Multiple Python Versions)

```bash
# Install pyenv using Homebrew
brew install pyenv
```

### 2. Install Python 3.11

```bash
# Install Python version 3.11.6
pyenv install 3.11.6
```

### 3. Select Python 3.11.6 Globally or Locally

```bash
# Set Python 3.11.6 as the global version
pyenv global 3.11.6

# OR set Python 3.11.6 for the specific project directory
cd /Users/pranav/projects/REA
pyenv local 3.11.6
```

### 4. Verify Python Version

```bash
# Check the currently active Python version
python --version
# Expected output: Python 3.11.6
```

### 5. Create a Fresh Virtual Environment

#### Remove Any Old Virtual Environment

```bash
# Navigate to the backend directory
cd /Users/pranav/projects/REA/backend

# Remove the existing virtual environment
rm -rf venv
```

#### Create & Activate a New Virtual Environment

```bash
# Create a new virtual environment using Python 3.11
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

#### Verify Python Version in Virtual Environment

```bash
# Check the Python version within the virtual environment
python --version
# Expected output: Python 3.11.x
```

### 6. Install Dependencies

#### Upgrade pip

```bash
# Upgrade pip to the latest version
pip install --upgrade pip
```

#### Install Requirements

If installing `torch` fails, install it first using the CPU-only index URL.

```bash
# Install torch and related packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies from requirements.txt
pip install -r requirements.txt

# OR install dependencies manually
pip install fastapi uvicorn requests pymongo faiss-cpu sentence-transformers
```

> **Important:** By default, `sentence-transformers` requires a compatible `torch` version. Since you’re on Apple Silicon with CPU-only, use the CPU index URL as shown above.

### 7. Run Your FastAPI App

Ensure you are in the `REA/backend/` directory or specify the correct module path if you are in the root folder.

```bash
# Run Uvicorn server from the backend directory
uvicorn app:app --host 0.0.0.0 --port 8000

# OR if you're one directory above backend
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

```


1. Ignore the Virtual Environment
Add venv/ or .venv/ to .gitignore:
gitignore
Copy
Edit
venv/
.venv/
Remove venv/ from Git tracking if previously committed:
bash
Copy
Edit
git rm -r --cached venv
git commit -m "Remove venv from tracking and update .gitignore"
This ensures that you’ll never commit large PyTorch libs or any other virtual environment files again.

2. Check for Large Files
If you previously committed large files (>100 MB) that are still in your Git history, GitHub will keep rejecting your pushes. To remove them:

Install BFG Repo-Cleaner:
bash
Copy
Edit
brew install bfg
Remove large files from history:
bash
Copy
Edit
bfg --delete-files "libtorch_cpu.dylib" .
Or remove the entire .venv folder from history:
bash
Copy
Edit
bfg --delete-folders venv .
Clean and force-push:
bash
Copy
Edit
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
Note: This rewrites your repo history, so coordinate with any collaborators.

3. Push the Clean Repo
Now that large files are removed from past commits, and your .venv/ is ignored in future commits:

Add & commit any new changes:
bash
Copy
Edit
git add .
git commit -m "My changes"
Push:
bash
Copy
Edit
git push origin main
You shouldn’t see file-size errors anymore.

