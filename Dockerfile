# 1. DOCKERFILE (create new file named "Dockerfile" - no extension)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

CMD ["python", "-u", "python_tg_ai_gemini.py"]

# 2. REQUIREMENTS.TXT (create new file)
python-telegram-bot==20.7
google-generativeai==0.3.2
httpx==0.25.2
beautifulsoup4==4.12.2
gtts==2.4.0
Pillow==10.1.0
PyPDF2==3.0.1

# 3. RAILWAY.JSON (create new file)
{
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "restartPolicyType": "ON_FAILURE",
    "sleepApplication": false
  }
}

# 4. .DOCKERIGNORE (create new file)
__pycache__/
*.pyc
*.log
.git
*.db
*.json
.DS_Store

# 5. CHANGES TO YOUR EXISTING PYTHON FILE
# Replace only the configuration section (around lines 60-80) with this:

# --- CONFIGURATION ---
import os

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/data/bot.log" if os.getenv('RAILWAY_ENVIRONMENT') else "bot.log"),
        logging.StreamHandler()
    ]
)

# Environment variables first, then config file fallback
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_KEYS_STR = os.getenv('GEMINI_API_KEYS')
ADMIN_IDS_STR = os.getenv('ADMIN_IDS', '')

if BOT_TOKEN and GEMINI_KEYS_STR:
    GEMINI_KEYS = [key.strip() for key in GEMINI_KEYS_STR.split(',')]
    ADMIN_IDS = [int(id.strip()) for id in ADMIN_IDS_STR.split(',') if id.strip().isdigit()]
    logging.info("Configuration loaded from environment variables")
else:
    try:
        with open("config.json", "r") as f: 
            config = json.load(f)
        BOT_TOKEN = config["TELEGRAM_BOT_TOKEN"]
        GEMINI_KEYS = config["GEMINI_API_KEYS"]
        ADMIN_IDS = config.get("ADMIN_IDS", [])
        logging.info("Configuration loaded from config.json")
    except (FileNotFoundError, KeyError) as e:
        logging.critical(f"FATAL: Configuration error: {e}")
        exit(1)

# Update these two lines only:
DB_FILE = "/app/data/bot_database.db" if os.getenv('RAILWAY_ENVIRONMENT') else "bot_database.db"
CACHE_FILE = "/app/data/response_cache.json" if os.getenv('RAILWAY_ENVIRONMENT') else "response_cache.json"

# 6. UPDATE setup_database() function - add this line at the beginning:
def setup_database():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)  # ADD THIS LINE
    conn = sqlite3.connect(DB_FILE)
    # ... rest of your existing code stays the same

# 7. UPDATE save_cache() method in SmartCache class:
def save_cache(self):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)  # ADD THIS LINE
        with open(CACHE_FILE, 'w') as f:
            json.dump(self.cache, f)
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")  # ADD ERROR HANDLING
