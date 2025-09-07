
#!/usr/bin/env python3
"""
ThinkByte-Mentor Bot: Einstein Edition v6.1 (Enhanced PDF Processing)

New Features in v6.1:
- Fixed PDF processing with proper integration
- Enhanced document handling for multiple formats
- Improved summarization capabilities
- Better error handling for document processing
- Knowledge base integration for uploaded documents
- Smart caching system for improved performance
- Enhanced error handling with retry mechanisms
- Rate limiting protection
- Advanced analytics and user insights
- Improved document parsing with multiple formats
- Dynamic learning reminders based on user activity
- Streak tracking and gamification elements
- Better memory management and conversation context
"""
import json
import os
import pathlib
import asyncio
import logging
import sqlite3
import random
import io
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

# Telegram and AI Imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters, 
    CallbackQueryHandler, ConversationHandler
)
import google.generativeai as genai

# Enhanced Feature Imports
import httpx
from bs4 import BeautifulSoup
from gtts import gTTS
import PIL.Image
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 not available - PDF support disabled")

# --- CONFIGURATION ---
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

try:
    with open("prompt.md", "r", encoding="utf-8") as f: 
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are ThinkByte, an advanced AI mentor focused on personalized learning and growth."

# Enhanced Constants
DB_FILE = "/app/data/bot_database.db" if os.getenv('RAILWAY_ENVIRONMENT') else "bot_database.db"
CACHE_FILE = "/app/data/response_cache.json" if os.getenv('RAILWAY_ENVIRONMENT') else "response_cache.json"
HISTORY_LENGTH = 15
MAX_RETRIES = 3
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 30
GET_NAME, GET_GOAL = range(2)

# --- ENHANCED CACHING SYSTEM ---
class SmartCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.load_cache()
    
    def _hash_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        hashed_key = self._hash_key(key)
        if hashed_key in self.cache:
            data, timestamp = self.cache[hashed_key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[hashed_key]
        return None
    
    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest]
        
        hashed_key = self._hash_key(key)
        self.cache[hashed_key] = (value, time.time())
        self.save_cache()
    
    def load_cache(self):
        try:
            with open(CACHE_FILE, 'r') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            self.cache = {}
    
    def save_cache(self):
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")

cache_system = SmartCache()

# --- RATE LIMITING ---
class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, user_id: int) -> bool:
        now = time.time()
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Clean old requests
        self.requests[user_id] = [req_time for req_time in self.requests[user_id] 
                                 if now - req_time < RATE_LIMIT_WINDOW]
        
        if len(self.requests[user_id]) < MAX_REQUESTS_PER_WINDOW:
            self.requests[user_id].append(now)
            return True
        return False

rate_limiter = RateLimiter()

# --- ENHANCED GEMINI API POOL ---
class GeminiPool:
    def __init__(self, keys):
        if not keys: raise ValueError("No Gemini API keys provided")
        self.keys = keys
        self.index = 0
        self.failed_keys = set()
    
    def get_key(self):
        attempts = 0
        while attempts < len(self.keys):
            key = self.keys[self.index]
            self.index = (self.index + 1) % len(self.keys)
            if key not in self.failed_keys:
                return key
            attempts += 1
        
        # Reset failed keys if all are failed
        self.failed_keys.clear()
        return self.keys[0]
    
    def mark_failed(self, key: str):
        self.failed_keys.add(key)

gemini_pool = GeminiPool(GEMINI_KEYS)

# --- ENHANCED DATABASE SETUP ---
def setup_database():
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Enhanced users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY, first_name TEXT, learning_goals TEXT,
            subscribed_to_tips BOOLEAN DEFAULT 0, last_checkin_ts DATETIME,
            streak_count INTEGER DEFAULT 0, total_messages INTEGER DEFAULT 0,
            preferred_language TEXT DEFAULT 'en', timezone TEXT DEFAULT 'UTC',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Enhanced messages table with metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, role TEXT, 
            content TEXT, response_time REAL, tokens_used INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    
    # Enhanced knowledge base
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, 
            source_name TEXT, content TEXT, content_hash TEXT,
            file_type TEXT, summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, content_hash)
        )
    """)
    
    # Analytics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
            event_type TEXT, event_data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

# --- ENHANCED DATABASE HELPERS ---
def get_user_profile(user_id: int) -> Optional[Tuple]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT first_name, learning_goals, streak_count, total_messages 
        FROM users WHERE user_id = ?
    """, (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def update_user_profile(user_id: int, **kwargs):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Insert or update user
    cursor.execute("""
        INSERT OR IGNORE INTO users (user_id, first_name) VALUES (?, ?)
    """, (user_id, kwargs.get('first_name', 'User')))
    
    for key, value in kwargs.items():
        if value is not None:
            cursor.execute(f"UPDATE users SET {key} = ? WHERE user_id = ?", (value, user_id))
    
    conn.commit()
    conn.close()

def get_user_history(user_id: int) -> List[Dict]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content FROM messages 
        WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
    """, (user_id, HISTORY_LENGTH))
    history = [{"role": row[0], "parts": [row[1]]} for row in cursor.fetchall()]
    conn.close()
    return list(reversed(history))

def add_message_to_history(user_id: int, first_name: str, role: str, 
                          content: str, response_time: float = 0):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR IGNORE INTO users (user_id, first_name) VALUES (?, ?)
    """, (user_id, first_name))
    
    cursor.execute("""
        INSERT INTO messages (user_id, role, content, response_time) 
        VALUES (?, ?, ?, ?)
    """, (user_id, role, content, response_time))
    
    # Update user stats
    cursor.execute("""
        UPDATE users SET total_messages = total_messages + 1 
        WHERE user_id = ?
    """, (user_id,))
    
    conn.commit()
    conn.close()

def log_analytics(user_id: int, event_type: str, event_data: str = ""):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analytics (user_id, event_type, event_data) VALUES (?, ?, ?)
    """, (user_id, event_type, event_data))
    conn.commit()
    conn.close()

# --- ENHANCED KNOWLEDGE BASE ---
def add_to_knowledge_base(user_id: int, source_name: str, content: str, 
                         file_type: str = "text", summary: str = ""):
    content_hash = hashlib.md5(content.encode()).hexdigest()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO knowledge_base 
        (user_id, source_name, content, content_hash, file_type, summary) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, source_name, content, content_hash, file_type, summary))
    conn.commit()
    conn.close()

def retrieve_from_knowledge_base(user_id: int, query: str) -> Optional[str]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT source_name, summary, content FROM knowledge_base 
        WHERE user_id = ? ORDER BY timestamp DESC
    """, (user_id,))
    documents = cursor.fetchall()
    conn.close()
    
    if not documents:
        return None
    
    query_words = set(query.lower().split())
    relevant_docs = []
    
    for name, summary, content in documents:
        search_text = f"{summary} {content}".lower()
        relevance_score = sum(1 for word in query_words if word in search_text)
        if relevance_score > 0:
            relevant_docs.append((relevance_score, name, summary or content[:200]))
    
    if not relevant_docs:
        return None
    
    # Sort by relevance and return top results
    relevant_docs.sort(reverse=True, key=lambda x: x[0])
    top_docs = relevant_docs[:3]
    
    return "\n---\n".join([f"📄 {name}:\n{summary}" for _, name, summary in top_docs])

def get_knowledge_base_list(user_id: int) -> List[Tuple[str, str, str]]:
    """Get list of user's knowledge base documents"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT source_name, file_type, timestamp FROM knowledge_base 
        WHERE user_id = ? ORDER BY timestamp DESC
    """, (user_id,))
    documents = cursor.fetchall()
    conn.close()
    return documents

def delete_from_knowledge_base(user_id: int, source_name: str) -> bool:
    """Delete a document from knowledge base"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM knowledge_base WHERE user_id = ? AND source_name = ?
    """, (user_id, source_name))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

# --- ENHANCED PDF PROCESSING ---
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2 with enhanced error handling."""
    if not PDF_SUPPORT:
        return "PDF reading not available. Please install PyPDF2: pip install PyPDF2"
    
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        
        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            return "This PDF is encrypted/password-protected. Please provide an unlocked version."
        
        # Get PDF info
        num_pages = len(pdf_reader.pages)
        logging.info(f"PDF has {num_pages} pages")
        
        text_content = []
        total_chars = 0
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                clean_text = page_text.strip()
                
                if clean_text and len(clean_text) > 20:  # Only meaningful content
                    text_content.append(f"=== Page {page_num + 1} ===\n{clean_text}")
                    total_chars += len(clean_text)
                    logging.info(f"Page {page_num + 1}: {len(clean_text)} chars extracted")
                else:
                    logging.warning(f"Page {page_num + 1}: Little or no text extracted")
                    
            except Exception as e:
                logging.error(f"Error extracting page {page_num + 1}: {e}")
                text_content.append(f"=== Page {page_num + 1} ===\n[Error extracting page: {e}]")
        
        full_text = "\n\n".join(text_content)
        logging.info(f"Total extracted: {total_chars} characters from {len(text_content)} pages")
        
        if total_chars < 50:  # Very little content extracted
            return f"Could not extract meaningful text from this PDF. It may contain images or have text embedded as graphics. Extracted only {total_chars} characters from {num_pages} pages."
        
        return full_text
    
    except Exception as e:
        logging.error(f"PDF processing error: {e}")
        return f"Error reading PDF: {e.__class__.__name__}: {str(e)}"

# --- ENHANCED WEB SEARCH ---
async def enhanced_web_search(query: str) -> str:
    cached_result = cache_system.get(f"search:{query}")
    if cached_result:
        return cached_result
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Try multiple search engines
            for search_url in [
                f"https://html.duckduckgo.com/html/?q={query}",
                f"https://www.startpage.com/sp/search?q={query}"
            ]:
                try:
                    response = await client.get(search_url, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        results = extract_search_results(soup, search_url)
                        if results:
                            cache_system.set(f"search:{query}", results)
                            return results
                except httpx.RequestError:
                    continue
        
        return "No search results found from available sources."
    except Exception as e:
        logging.error(f"Web search failed: {e}")
        return f"Search temporarily unavailable: {e.__class__.__name__}"

def extract_search_results(soup: BeautifulSoup, url: str) -> str:
    """Extract search results from BeautifulSoup object."""
    results = []
    if "duckduckgo" in url:
        for result in soup.find_all('a', class_='result__a')[:5]:
            title = result.get_text(strip=True)
            if title:
                results.append(f"• {title}")
    else:  # Startpage
        for result in soup.find_all('h3', class_='search-result-title')[:5]:
            title = result.get_text(strip=True)
            if title:
                results.append(f"• {title}")
    
    return "\n".join(results) if results else ""

# --- ENHANCED CODE EXECUTION ---
async def execute_code_with_retry(language: str, code: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post("https://emkc.org/api/v2/piston/execute", 
                    json={
                        "language": language, 
                        "version": "*", 
                        "files": [{"content": code}]
                    })
                
                if response.status_code == 200:
                    result = response.json()
                    if "run" in result:
                        output = result["run"].get("output", "")
                        stderr = result["run"].get("stderr", "")
                        if output:
                            return f"✅ **Output:**\n```\n{output[:1500]}\n```"
                        elif stderr:
                            return f"⚠️ **Error:**\n```\n{stderr[:1000]}\n```"
                        else:
                            return "✅ Code executed successfully (no output)"
                    else:
                        return f"❌ Execution failed: {result.get('message', 'Unknown error')}"
                
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return f"❌ Code execution failed after {MAX_RETRIES} attempts: {e.__class__.__name__}"
            await asyncio.sleep(1)  # Brief delay before retry

# --- DOCUMENT SUMMARIZATION ---
async def generate_summary(user_id: int, content: str, file_name: str) -> str:
    """Generate a summary of document content using AI"""
    try:
        summary_prompt = f"Please create a brief summary (2-3 sentences) of this document titled '{file_name}':\n\n{content[:2000]}"
        summary = await get_gemini_response(user_id, summary_prompt)
        return summary[:300]  # Limit summary length
    except Exception as e:
        logging.error(f"Summary generation failed: {e}")
        return f"Document uploaded: {file_name} ({len(content)} characters)"

# --- ENHANCED AI RESPONSE GENERATION ---
async def get_gemini_response(user_id: int, user_prompt: str, 
                            image: PIL.Image.Image = None, 
                            document_text: str = None) -> str:
    start_time = time.time()
    logging.info(f"Getting AI response for user {user_id}: {user_prompt[:50]}...")
    
    # Check cache first for text-only queries
    if not image and not document_text:
        cached_response = cache_system.get(f"ai:{user_id}:{user_prompt}")
        if cached_response:
            logging.info("Returning cached response")
            return cached_response
    
    history = get_user_history(user_id)
    profile = get_user_profile(user_id)
    
    # Enhanced personalization - add to user prompt instead of system instruction
    enhanced_prompt = user_prompt
    if profile:
        name, goal, streak, total_msgs = profile
        enhanced_prompt = f"{SYSTEM_PROMPT}\n\nUser Profile: {name}, Goal: {goal}, Streak: {streak} days\n\nUser: {user_prompt}"
    else:
        enhanced_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}"
    
    # Enhanced RAG
    knowledge_context = retrieve_from_knowledge_base(user_id, user_prompt)
    if knowledge_context:
        enhanced_prompt = f"{SYSTEM_PROMPT}\n\nKnowledge Base Context:\n{knowledge_context}\n\nUser Question: {user_prompt}"
    elif any(k in user_prompt.lower() for k in ["latest", "current", "recent", "today", "2025"]):
        search_results = await enhanced_web_search(user_prompt)
        if search_results and "unavailable" not in search_results:
            enhanced_prompt = f"{SYSTEM_PROMPT}\n\nWeb Results:\n{search_results}\n\nUser Question: {user_prompt}"
    
    content_parts = [enhanced_prompt]
    if image:
        content_parts.append(image)
    if document_text:
        content_parts.insert(0, f"{SYSTEM_PROMPT}\n\nDocument to analyze:\n{document_text[:8000]}\n\nUser request: {user_prompt}")
        content_parts = content_parts[:1]  # Keep only the combined prompt
    
    # Try API call with retry logic
    for attempt in range(MAX_RETRIES):
        api_key = gemini_pool.get_key()
        logging.info(f"Attempt {attempt + 1} with API key ...{api_key[-4:]}")
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            
            logging.info("Sending request to Gemini API...")
            response = await model.generate_content_async(
                contents=history + [{"role": "user", "parts": content_parts}]
            )
            
            response_time = time.time() - start_time
            response_text = response.text
            logging.info(f"Got response from Gemini in {response_time:.2f}s")
            
            # Cache successful responses
            if not image and not document_text:
                cache_system.set(f"ai:{user_id}:{user_prompt}", response_text)
            
            return response_text
            
        except Exception as e:
            logging.error(f"Gemini API error on attempt {attempt + 1}: {e}")
            gemini_pool.mark_failed(api_key)
            if attempt == MAX_RETRIES - 1:
                return f"I'm having trouble connecting to my AI brain right now. Please try again in a moment.\n\nTechnical details: {e.__class__.__name__}: {str(e)[:100]}"
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# --- ENHANCED HANDLERS ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    log_analytics(user.id, "command_used", "/start")
    
    profile = get_user_profile(user.id)
    if profile and profile[1]:  # Has learning goal
        welcome_msg = f"Welcome back, {profile[0]}! 🎯\n\n"
        welcome_msg += f"**Your Goal:** {profile[1]}\n"
        welcome_msg += f"**Current Streak:** {profile[2]} days 🔥\n"
        welcome_msg += f"**Total Messages:** {profile[3]} 💬\n\n"
        welcome_msg += "How can I help you learn today? Use /help for all commands."
        await update.message.reply_text(welcome_msg)
        return ConversationHandler.END
    else:
        update_user_profile(user.id, first_name=user.first_name)
        await update.message.reply_text(
            f"🚀 Welcome to ThinkByte, {user.first_name}!\n\n"
            "I'm your AI learning mentor. Let's personalize your experience.\n\n"
            "What should I call you? (or /cancel to skip setup)"
        )
        return GET_NAME

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logging.info(f"Received message from {user.first_name} ({user.id}): {update.message.text}")
    
    if not rate_limiter.is_allowed(user.id):
        await update.message.reply_text(
            "⏱️ You're sending messages too quickly. Please wait a moment."
        )
        return
    
    prompt = update.message.text
    await context.bot.send_chat_action(chat_id=user.id, action="typing")
    
    start_time = time.time()
    add_message_to_history(user.id, user.first_name, "user", prompt)
    
    try:
        logging.info("Getting AI response...")
        response_text = await get_gemini_response(user.id, prompt)
        response_time = time.time() - start_time
        
        logging.info(f"AI response received in {response_time:.2f}s")
        add_message_to_history(user.id, user.first_name, "model", response_text, response_time)
        log_analytics(user.id, "message_processed", f"response_time:{response_time:.2f}")
        
        # Parse follow-ups and create buttons
        main_text, follow_ups = parse_follow_ups(response_text)
        buttons = [InlineKeyboardButton(q, callback_data=q) for q in follow_ups[:3]]
        reply_markup = InlineKeyboardMarkup([buttons]) if buttons else None
        
        # Send response
        await update.message.reply_text(main_text, reply_markup=reply_markup)
        
    except Exception as e:
        logging.error(f"Error in handle_text_message: {e}")
        await update.message.reply_text(
            "Sorry, I encountered an error processing your message. Please try again."
        )

def parse_follow_ups(text: str) -> Tuple[str, List[str]]:
    lines = text.split('\n')
    follow_up_markers = ["follow-up:", "follow up:", "next questions:", "related:"]
    
    for i, line in enumerate(lines):
        if any(marker in line.lower() for marker in follow_up_markers):
            main_text = "\n".join(lines[:i]).strip()
            follow_up_text = "\n".join(lines[i:])
            # Extract questions from follow-up text
            questions = []
            for q_line in follow_up_text.split('\n')[1:]:  # Skip marker line
                cleaned = q_line.strip().lstrip('-•').strip()
                if cleaned and len(cleaned) < 100:  # Reasonable length
                    questions.append(cleaned)
            return main_text, questions[:3]  # Limit to 3 questions
    
    return text, []

# --- CONVERSATION HANDLERS ---
async def received_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    name = update.message.text
    update_user_profile(user.id, first_name=name)
    await update.message.reply_text(
        f"Perfect, {name}! 🎯\n\n"
        "What is your main learning goal right now?\n"
        "(e.g., 'become a Python developer', 'learn data science', 'start freelancing')"
    )
    return GET_GOAL

async def received_goal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    goal = update.message.text
    update_user_profile(user.id, learning_goals=goal)
    await update.message.reply_text(
        f"✅ Perfect! I'll tailor my advice to help you '{goal}'.\n\n"
        "🚀 You're all set! Here's what I can help you with:\n"
        "• Ask me anything about your learning journey\n"
        "• Send photos of code/diagrams for analysis\n"
        "• Upload documents (including PDFs) for analysis\n"
        "• Use /help to see all available commands\n\n"
        "What would you like to learn about first?"
    )
    return ConversationHandler.END

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Setup cancelled. You can restart it anytime with /start."
    )
    return ConversationHandler.END

# --- ENHANCED DOCUMENT HANDLER ---
async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    prompt = update.message.caption or "Please analyze this document and summarize its key points."
    
    await context.bot.send_chat_action(chat_id=user.id, action="upload_document")
    
    try:
        doc_file = await update.message.document.get_file()
        doc_bytes = await doc_file.download_as_bytearray()
        file_name = update.message.document.file_name or "document"
        file_extension = file_name.split('.')[-1].lower() if '.' in file_name else ''
        
        document_text = ""
        
        # Handle PDF files
        if file_extension == 'pdf':
            if not PDF_SUPPORT:
                await update.message.reply_text(
                    "📄 PDF support is not available. Please install PyPDF2:\n"
                    "`pip install PyPDF2`\n\n"
                    "Or convert your PDF to a text file (.txt) and try again."
                )
                return
            
            # Show processing message
            processing_msg = await update.message.reply_text("📄 Processing PDF, please wait...")
            
            # Process PDF
            document_text = extract_pdf_text(doc_bytes)
            
            # Delete processing message
            await processing_msg.delete()
            
            # Check if extraction was successful
            if document_text.startswith("Error") or document_text.startswith("This PDF is encrypted") or document_text.startswith("Could not extract"):
                await update.message.reply_text(f"❌ **PDF Processing Failed**\n\n{document_text}")
                return
                
        # Handle text-based files
        elif file_extension in ['txt', 'py', 'md', 'json', 'csv', 'log', 'yml', 'yaml', 'xml', 'js', 'html', 'css']:
            try:
                document_text = doc_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    document_text = doc_bytes.decode('latin-1')
                except UnicodeDecodeError:
                    await update.message.reply_text(
                        f"❌ Could not decode '{file_name}' as text.\n"
                        "Supported formats: PDF, TXT, PY, MD, JSON, CSV, LOG, YML, XML"
                    )
                    return
        else:
            await update.message.reply_text(
                f"❌ Unsupported file type: `.{file_extension}`\n\n"
                "**Supported formats:**\n"
                "• PDF files (.pdf)\n"
                "• Text files (.txt, .md, .log)\n"
                "• Code files (.py, .js, .html, .css)\n"
                "• Data files (.json, .csv, .yml, .xml)"
            )
            return
        
        # Check if we got meaningful content
        if not document_text or len(document_text.strip()) < 10:
            await update.message.reply_text(
                f"❌ No readable content found in '{file_name}'\n"
                "The file might be empty, corrupted, or contain only images."
            )
            return
        
        # Add to message history
        add_message_to_history(user.id, user.first_name, "user", f"[Document: {file_name}] {prompt}")
        
        # Truncate if too long and process with AI
        original_length = len(document_text)
        truncated_text = document_text[:8000]
        
        if original_length > 8000:
            prompt += f"\n\n(Note: Document was truncated from {original_length:,} to 8,000 characters for processing)"
        
        # Generate AI response
        response_text = await get_gemini_response(user.id, prompt, document_text=truncated_text)
        add_message_to_history(user.id, user.first_name, "model", response_text)
        
        # Generate summary for knowledge base
        summary = await generate_summary(user.id, document_text[:2000], file_name)
        
        # Add to knowledge base
        add_to_knowledge_base(user.id, file_name, document_text, file_extension, summary)
        
        log_analytics(user.id, "document_analysis", f"{file_extension}:{len(document_text)}")
        
        # Send response with option to save
        buttons = [
            InlineKeyboardButton("📚 View Knowledge Base", callback_data="view_kb"),
            InlineKeyboardButton("🗑️ Remove from KB", callback_data=f"remove_kb:{file_name}")
        ]
        reply_markup = InlineKeyboardMarkup([buttons])
        
        await update.message.reply_text(
            f"✅ **Document Processed: {file_name}**\n\n{response_text}\n\n"
            f"📊 *Added to your knowledge base ({len(document_text):,} characters)*",
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logging.error(f"Document processing failed: {e}")
        await update.message.reply_text("Sorry, I couldn't process that document. Please try again.")

# --- PHOTO AND OTHER HANDLERS ---
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    prompt = update.message.caption or "Please analyze this image and explain what you see."
    
    await context.bot.send_chat_action(chat_id=user.id, action="upload_photo")
    
    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        image = PIL.Image.open(io.BytesIO(image_bytes))
        
        add_message_to_history(user.id, user.first_name, "user", f"[Image] {prompt}")
        response_text = await get_gemini_response(user.id, prompt, image=image)
        add_message_to_history(user.id, user.first_name, "model", response_text)
        
        log_analytics(user.id, "image_analysis", "photo")
        await update.message.reply_text(response_text)
        
    except Exception as e:
        logging.error(f"Photo processing failed: {e}")
        await update.message.reply_text("Sorry, I couldn't process that image. Please try again.")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    
    if query.data == "view_kb":
        documents = get_knowledge_base_list(user_id)
        if documents:
            kb_text = "📚 **Your Knowledge Base:**\n\n"
            for name, file_type, timestamp in documents:
                kb_text += f"• {name} (.{file_type}) - {timestamp.split()[0]}\n"
            await query.edit_message_text(kb_text)
        else:
            await query.edit_message_text("Your knowledge base is empty. Upload documents to start building it!")
    
    elif query.data.startswith("remove_kb:"):
        file_name = query.data.split(":", 1)[1]
        if delete_from_knowledge_base(user_id, file_name):
            await query.edit_message_text(f"✅ Removed '{file_name}' from your knowledge base.")
        else:
            await query.edit_message_text(f"❌ Could not find '{file_name}' in your knowledge base.")
    
    else:
        # Handle follow-up questions
        await query.edit_message_reply_markup(reply_markup=None)
        new_text = f"➡️ {query.data}"
        message = await query.message.reply_text(new_text)
        
        fake_update = Update(
            update_id=update.update_id,
            message=message
        )
        fake_update.message.text = query.data
        fake_update.effective_user = query.from_user
        
        await handle_text_message(fake_update, context)

# --- ADDITIONAL COMMANDS ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """🤖 **ThinkByte Bot v6.1 - Your AI Learning Mentor**

**🧠 Core Features**
• **Chat**: Send any message for AI assistance
• **Photo Analysis**: Send images of code, diagrams, etc.
• **Document Upload**: Upload PDFs, text files for analysis

**📚 Knowledge Management**
• `/teach` - Reply to a document to add it to your knowledge base
• `/knowledge` - View your saved documents
• `/forget <name>` - Remove a document from memory

**👤 Profile & Progress**
• `/profile` - View your learning profile and goals
• `/stats` - See your usage statistics and streak
• `/reset` - Clear conversation history

**🛠️ Productivity Tools**
• `/run <language>\\n<code>` - Execute code snippets
• `/speak <text>` - Convert text to voice message

**⚙️ Bot Management**
• `/debug` - System diagnostics
• `/help` - Show this help message

**Supported File Types:**
PDF, TXT, PY, MD, JSON, CSV, LOG, YML, XML, JS, HTML, CSS

Ready to learn? Just send me a message!"""
    
    await update.message.reply_text(help_text)

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    profile = get_user_profile(user_id)
    
    if profile and profile[1]:
        name, goal, streak, total_msgs = profile
        profile_text = f"👤 **Your Learning Profile**\n\n"
        profile_text += f"**Name:** {name}\n"
        profile_text += f"**Goal:** {goal}\n"
        profile_text += f"**Current Streak:** {streak} days\n"
        profile_text += f"**Total Messages:** {total_msgs}\n\n"
        profile_text += "Use /start to update your profile!"
        
        await update.message.reply_text(profile_text)
    else:
        await update.message.reply_text(
            "I don't have a complete profile for you yet.\n"
            "Use /start to set up your personalized learning experience!"
        )

async def knowledge_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    documents = get_knowledge_base_list(user_id)
    
    if documents:
        kb_text = "📚 **Your Knowledge Base:**\n\n"
        for name, file_type, timestamp in documents:
            kb_text += f"• {name} (.{file_type}) - {timestamp.split()[0]}\n"
        kb_text += f"\n📊 Total: {len(documents)} documents"
        await update.message.reply_text(kb_text)
    else:
        await update.message.reply_text("Your knowledge base is empty. Upload documents to start building it!")

async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /forget <document_name>")
        return
    
    doc_name = " ".join(context.args)
    if delete_from_knowledge_base(user_id, doc_name):
        await update.message.reply_text(f"✅ Removed '{doc_name}' from your knowledge base.")
    else:
        await update.message.reply_text(f"❌ Could not find '{doc_name}' in your knowledge base.")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE user_id = ?", (update.effective_user.id,))
    conn.commit()
    conn.close()
    
    log_analytics(update.effective_user.id, "command_used", "/reset")
    await update.message.reply_text("✅ Conversation history cleared successfully!")

async def run_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        parts = update.message.text.split('\n', 1)
        if len(parts) < 2:
            raise ValueError("Missing code")
        
        lang = parts[0].split(' ', 1)[1].strip()
        code = parts[1].strip()
        
        if not lang or not code:
            raise ValueError("Missing language or code")
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        log_analytics(update.effective_user.id, "code_execution", lang)
        result = await execute_code_with_retry(lang, code)
        await update.message.reply_text(result)
        
    except (IndexError, ValueError):
        await update.message.reply_text(
            "**Usage:** `/run <language>\\n<code>`\n\n"
            "**Example:**\n"
            "`/run python`\n"
            "`print('Hello, World!')`"
        )

async def speak_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_to_speak = " ".join(context.args)
    if not text_to_speak:
        await update.message.reply_text("**Usage:** `/speak <text>`")
        return
    
    if len(text_to_speak) > 500:
        await update.message.reply_text("Text too long! Please limit to 500 characters.")
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_voice")
    
    try:
        tts = gTTS(text=text_to_speak, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        
        log_analytics(update.effective_user.id, "voice_synthesis", str(len(text_to_speak)))
        await update.message.reply_voice(voice=audio_fp)
        
    except Exception as e:
        logging.error(f"TTS failed: {e}")
        await update.message.reply_text("Sorry, I couldn't generate audio right now.")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # User stats
    cursor.execute("""
        SELECT total_messages, streak_count, created_at 
        FROM users WHERE user_id = ?
    """, (user_id,))
    user_stats = cursor.fetchone()
    
    # Message stats
    cursor.execute("""
        SELECT COUNT(*), AVG(response_time) 
        FROM messages WHERE user_id = ? AND role = 'model'
    """, (user_id,))
    msg_stats = cursor.fetchone()
    
    # Knowledge base stats
    cursor.execute("""
        SELECT COUNT(*) FROM knowledge_base WHERE user_id = ?
    """, (user_id,))
    kb_count = cursor.fetchone()[0]
    
    conn.close()
    
    if user_stats:
        total_msgs, streak, created = user_stats
        avg_response = msg_stats[1] or 0
        
        stats_text = f"📊 **Your ThinkByte Stats**\n\n"
        stats_text += f"🎯 **Current Streak:** {streak} days\n"
        stats_text += f"💬 **Total Messages:** {total_msgs}\n"
        stats_text += f"📚 **Knowledge Base Items:** {kb_count}\n"
        stats_text += f"⚡ **Avg Response Time:** {avg_response:.1f}s\n"
        stats_text += f"📅 **Member Since:** {created.split()[0]}\n"
        
        await update.message.reply_text(stats_text)
    else:
        await update.message.reply_text("No stats available yet. Start chatting to build your profile!")

async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if ADMIN_IDS and user_id not in ADMIN_IDS:
        await update.message.reply_text("This command is restricted to administrators.")
        return
    
    await update.message.reply_text("🔍 Running system diagnostics...")
    
    keys_status = f"✅ {len(GEMINI_KEYS)} Gemini API key(s) loaded" if GEMINI_KEYS else "❌ No API keys found"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get("https://8.8.8.8")
        internet_status = "✅ Internet connection OK"
    except Exception as e:
        internet_status = f"❌ Internet connection failed: {e.__class__.__name__}"
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        conn.close()
        db_status = f"✅ Database OK ({user_count} users)"
    except Exception as e:
        db_status = f"❌ Database error: {e.__class__.__name__}"
    
    diagnostics = f"""🩺 **System Diagnostics**

{keys_status}
{internet_status}
{db_status}

**Cache:** {len(cache_system.cache)} items
**Failed Keys:** {len(gemini_pool.failed_keys)}
**PDF Support:** {"✅ Available" if PDF_SUPPORT else "❌ Not available"}"""
    
    await update.message.reply_text(diagnostics)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Exception while handling an update: {context.error}")
    
    if update and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Something went wrong. Please try again in a moment."
            )
        except Exception:
            pass

# --- MAIN APPLICATION SETUP ---
def main():
    setup_database()
    
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .connection_pool_size(8)
        .pool_timeout(20)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(10)
        .get_updates_connection_pool_size(1)
        .get_updates_pool_timeout(10)
        .get_updates_read_timeout(10)
        .get_updates_write_timeout(10)
        .get_updates_connect_timeout(5)
        .build()
    )
    
    application.add_error_handler(error_handler)
    
    # Conversation handler for onboarding
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            GET_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_name)],
            GET_GOAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_goal)],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
    )
    
    # Add handlers
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("knowledge", knowledge_command))
    application.add_handler(CommandHandler("forget", forget_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("run", run_command))
    application.add_handler(CommandHandler("speak", speak_command))
    application.add_handler(CommandHandler("debug", debug_command))
    
    # Message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document_message))
    application.add_handler(CallbackQueryHandler(button_handler))
    
    logging.info("🤖 ThinkByte Bot v6.1 starting with enhanced PDF processing...")
    
    try:
        application.run_polling(
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"]
        )
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Bot crashed: {e}")
        logging.info("Restarting in 5 seconds...")
        time.sleep(5)
        main()

if __name__ == "__main__":
    main()
