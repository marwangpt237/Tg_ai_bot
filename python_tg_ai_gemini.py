
#!/usr/bin/env python3
"""
ThinkByte-Mentor Bot: Einstein Edition v6.2 (Enhanced Educational Features)
"""
import json, os, asyncio, logging, sqlite3, io, hashlib, time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler, ConversationHandler
import google.generativeai as genai
import httpx
from bs4 import BeautifulSoup
from gtts import gTTS
import PIL.Image
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# CONFIGURATION
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("/app/logs/bot.log" if os.getenv('RAILWAY_ENVIRONMENT') else "bot.log"),
                              logging.StreamHandler()])

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_KEYS_STR = os.getenv('GEMINI_API_KEYS')
ADMIN_IDS_STR = os.getenv('ADMIN_IDS', '')

if BOT_TOKEN and GEMINI_KEYS_STR:
    GEMINI_KEYS = [key.strip() for key in GEMINI_KEYS_STR.split(',')]
    ADMIN_IDS = [int(id.strip()) for id in ADMIN_IDS_STR.split(',') if id.strip().isdigit()]
else:
    try:
        with open("config.json", "r") as f: 
            config = json.load(f)
        BOT_TOKEN = config["TELEGRAM_BOT_TOKEN"]
        GEMINI_KEYS = config["GEMINI_API_KEYS"]
        ADMIN_IDS = config.get("ADMIN_IDS", [])
    except (FileNotFoundError, KeyError) as e:
        logging.critical(f"Configuration error: {e}")
        exit(1)

try:
    with open("prompt.md", "r", encoding="utf-8") as f: 
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are ThinkByte, an AI mentor for young Algerians learning freelancing, programming, and marketing."

DB_FILE = "/app/data/bot_database.db" if os.getenv('RAILWAY_ENVIRONMENT') else "bot_database.db"
CACHE_FILE = "/app/data/response_cache.json" if os.getenv('RAILWAY_ENVIRONMENT') else "response_cache.json"
GET_NAME, GET_GOAL = range(2)

class SmartCache:
    def __init__(self):
        self.cache = {}
        self.load_cache()
    
    def _hash_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        hashed_key = self._hash_key(key)
        if hashed_key in self.cache:
            return self.cache[hashed_key][0]
        return None
    
    def set(self, key: str, value: str):
        self.cache[self._hash_key(key)] = (value, time.time())
        self.save_cache()
    
    def load_cache(self):
        try:
            with open(CACHE_FILE, 'r') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            pass
    
    def save_cache(self):
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f)
        except: pass

class GeminiPool:
    def __init__(self, keys):
        self.keys = keys
        self.index = 0
        self.failed = set()
    
    def get_key(self):
        for _ in range(len(self.keys)):
            key = self.keys[self.index]
            self.index = (self.index + 1) % len(self.keys)
            if key not in self.failed:
                return key
        self.failed.clear()
        return self.keys[0]
    
    def mark_failed(self, key: str):
        self.failed.add(key)

cache_system = SmartCache()
gemini_pool = GeminiPool(GEMINI_KEYS)

def setup_database():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY, first_name TEXT, learning_goals TEXT,
        streak_count INTEGER DEFAULT 0, total_messages INTEGER DEFAULT 0,
        last_activity_date DATE, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, role TEXT, 
        content TEXT, response_time REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, 
        source_name TEXT, content TEXT, content_hash TEXT, summary TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_id, content_hash))""")
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        event_type TEXT, event_data TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    
    # Phase 1 Tables
    cursor.execute("""CREATE TABLE IF NOT EXISTS quizzes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        document_name TEXT, score INTEGER, total_questions INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS daily_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        date DATE, activity_count INTEGER DEFAULT 0, UNIQUE(user_id, date))""")
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS resources (
        id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, title TEXT, url TEXT,
        description TEXT, skill_level TEXT, added_by INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    
    conn.commit()
    conn.close()

def get_user_profile(user_id: int):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT first_name, learning_goals, streak_count, total_messages FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def update_user_profile(user_id: int, **kwargs):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users (user_id, first_name) VALUES (?, ?)", (user_id, kwargs.get('first_name', 'User')))
    for key, value in kwargs.items():
        if value is not None:
            cursor.execute(f"UPDATE users SET {key} = ? WHERE user_id = ?", (value, user_id))
    conn.commit()
    conn.close()

def add_message_to_history(user_id: int, first_name: str, role: str, content: str, response_time: float = 0):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users (user_id, first_name) VALUES (?, ?)", (user_id, first_name))
    cursor.execute("INSERT INTO messages (user_id, role, content, response_time) VALUES (?, ?, ?, ?)", 
                   (user_id, role, content, response_time))
    cursor.execute("UPDATE users SET total_messages = total_messages + 1 WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def log_analytics(user_id: int, event_type: str, event_data: str = ""):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO analytics (user_id, event_type, event_data) VALUES (?, ?, ?)", (user_id, event_type, event_data))
    conn.commit()
    conn.close()

# PHASE 1 FEATURES
def update_streak(user_id: int):
    today = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""INSERT OR REPLACE INTO daily_progress (user_id, date, activity_count)
        VALUES (?, ?, COALESCE((SELECT activity_count FROM daily_progress WHERE user_id = ? AND date = ?), 0) + 1)""",
        (user_id, today, user_id, today))
    cursor.execute("UPDATE users SET streak_count = streak_count + 1, last_activity_date = ? WHERE user_id = ? AND (last_activity_date != ? OR last_activity_date IS NULL)",
        (today, user_id, today))
    conn.commit()
    conn.close()

def add_to_knowledge_base(user_id: int, source_name: str, content: str, summary: str = ""):
    content_hash = hashlib.md5(content.encode()).hexdigest()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO knowledge_base (user_id, source_name, content, content_hash, summary) VALUES (?, ?, ?, ?, ?)",
        (user_id, source_name, content, content_hash, summary))
    conn.commit()
    conn.close()

def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not PDF_SUPPORT:
        return "PDF support unavailable. Install PyPDF2."
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        if pdf_reader.is_encrypted:
            return "PDF is encrypted."
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        return text if len(text) > 50 else "Could not extract meaningful text from PDF."
    except Exception as e:
        return f"PDF processing error: {e}"

async def get_gemini_response(user_id: int, prompt: str, image=None, document_text=None) -> str:
    cached = cache_system.get(f"ai:{user_id}:{prompt}") if not image and not document_text else None
    if cached:
        return cached
    
    enhanced_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}"
    if document_text:
        enhanced_prompt = f"{SYSTEM_PROMPT}\n\nDocument:\n{document_text[:4000]}\n\nUser: {prompt}"
    
    content_parts = [enhanced_prompt]
    if image:
        content_parts.append(image)
    
    for attempt in range(3):
        try:
            genai.configure(api_key=gemini_pool.get_key())
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = await model.generate_content_async(content_parts)
            result = response.text
            if not image and not document_text:
                cache_system.set(f"ai:{user_id}:{prompt}", result)
            return result
        except Exception as e:
            if attempt == 2:
                return f"AI service temporarily unavailable: {str(e)[:100]}"

# COMMAND HANDLERS
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    profile = get_user_profile(user.id)
    if profile and profile[1]:
        await update.message.reply_text(f"Welcome back, {profile[0]}!\nGoal: {profile[1]}\nStreak: {profile[2]} days\n\nHow can I help you learn today?")
        return ConversationHandler.END
    else:
        update_user_profile(user.id, first_name=user.first_name)
        await update.message.reply_text(f"Welcome to ThinkByte, {user.first_name}!\nWhat should I call you?")
        return GET_NAME

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    prompt = update.message.text
    
    start_time = time.time()
    add_message_to_history(user.id, user.first_name, "user", prompt)
    
    try:
        response_text = await get_gemini_response(user.id, prompt)
        response_time = time.time() - start_time
        add_message_to_history(user.id, user.first_name, "model", response_text, response_time)
        update_streak(user.id)
        await update.message.reply_text(response_text)
    except Exception as e:
        await update.message.reply_text("Sorry, I encountered an error. Please try again.")

async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    prompt = update.message.caption or "Please analyze this document."
    
    try:
        doc_file = await update.message.document.get_file()
        doc_bytes = await doc_file.download_as_bytearray()
        file_name = update.message.document.file_name or "document"
        file_ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
        
        if file_ext == 'pdf':
            document_text = extract_pdf_text(doc_bytes)
        elif file_ext in ['txt', 'py', 'md', 'json', 'csv']:
            document_text = doc_bytes.decode('utf-8', errors='ignore')
        else:
            await update.message.reply_text("Unsupported file type. Use PDF, TXT, PY, MD, JSON, or CSV.")
            return
        
        if len(document_text.strip()) < 10:
            await update.message.reply_text("No readable content found.")
            return
        
        response_text = await get_gemini_response(user.id, prompt, document_text=document_text)
        summary = response_text[:200] + "..." if len(response_text) > 200 else response_text
        add_to_knowledge_base(user.id, file_name, document_text, summary)
        
        await update.message.reply_text(f"Document processed: {file_name}\n\n{response_text}")
        log_analytics(user.id, "document_processed", file_ext)
    except Exception as e:
        await update.message.reply_text("Failed to process document. Please try again.")

async def quiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT source_name, content FROM knowledge_base WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        await update.message.reply_text("Upload a document first to generate a quiz!")
        return
    
    doc_name, content = result
    quiz_prompt = f"Create 3 multiple choice questions from {doc_name}:\n\n{content[:1000]}"
    quiz = await get_gemini_response(user_id, quiz_prompt)
    await update.message.reply_text(f"Quiz from {doc_name}:\n\n{quiz}")

async def streak_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    profile = get_user_profile(update.effective_user.id)
    if profile:
        await update.message.reply_text(f"Learning Streak: {profile[2]} days\nTotal Messages: {profile[3]}\nGoal: {profile[1]}")
    else:
        await update.message.reply_text("Start with /start first!")

async def resources_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT category, title, url, skill_level FROM resources ORDER BY category")
    resources = cursor.fetchall()
    conn.close()
    
    if not resources:
        await update.message.reply_text("No resources available yet!")
        return
    
    text = "Learning Resources:\n\n"
    for category, title, url, level in resources:
        text += f"• {title} ({level}) - {url}\n"
    await update.message.reply_text(text)

async def admin_add_resource(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    if len(context.args) < 5:
        await update.message.reply_text("Usage: /add_resource <category> <title> <url> <description> <level>")
        return
    
    category, title, url, description, level = context.args[:5]
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO resources (category, title, url, description, skill_level, added_by) VALUES (?, ?, ?, ?, ?, ?)",
        (category, title, url, description, level, update.effective_user.id))
    conn.commit()
    conn.close()
    await update.message.reply_text(f"Added resource: {title}")

async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    users = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM messages WHERE timestamp >= date('now', '-7 days')")
    messages = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    docs = cursor.fetchone()[0]
    conn.close()
    
    await update.message.reply_text(f"Bot Stats:\nUsers: {users}\nMessages (7d): {messages}\nDocuments: {docs}")

async def received_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.message.text
    update_user_profile(update.effective_user.id, first_name=name)
    await update.message.reply_text(f"Great, {name}! What's your learning goal?\n(e.g., 'learn programming', 'start freelancing')")
    return GET_GOAL

async def received_goal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    goal = update.message.text
    update_user_profile(update.effective_user.id, learning_goals=goal)
    await update.message.reply_text(f"Perfect! I'll help you '{goal}'.\n\nSend me any question or upload documents to get started!")
    return ConversationHandler.END

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """ThinkByte Bot - AI Learning Mentor

Core Commands:
• /start - Setup profile
• /help - Show commands
• /quiz - Generate quiz from documents
• /streak - View learning progress
• /resources - Browse learning materials

Features:
• Upload PDFs, text files for analysis
• Send photos for image analysis
• Ask questions in English or Arabic
• Track daily learning streaks

Start learning now!"""
    await update.message.reply_text(help_text)

def main():
    setup_database()
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            GET_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_name)],
            GET_GOAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_goal)],
        },
        fallbacks=[],
    )
    
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("quiz", quiz_command))
    application.add_handler(CommandHandler("streak", streak_command))
    application.add_handler(CommandHandler("resources", resources_command))
    application.add_handler(CommandHandler("add_resource", admin_add_resource))
    application.add_handler(CommandHandler("admin_stats", admin_stats))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document_message))
    
    logging.info("ThinkByte Bot v6.2 starting...")
    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()