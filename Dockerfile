# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data

# Run the bot
CMD ["python", "python_tg_ai_gemini.py"]



# requirements.txt
python-telegram-bot==20.7
google-generativeai==0.3.2
httpx==0.25.2
beautifulsoup4==4.12.2
gtts==2.4.0
Pillow==10.1.0
PyPDF2==3.0.1



# railway.json
{
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "restartPolicyType": "ON_FAILURE",
    "sleepApplication": false
  }
}
