import discord
from discord.ext import commands
import os
import sqlite3
from huggingface_hub import InferenceClient

# Load tokens from env
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face client
client_hf = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

# Intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- SQLite Memory ---
DB_FILE = "chat_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        msg TEXT
    )
    """)
    conn.commit()
    conn.close()

def add_message(user, msg):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO memory (user, msg) VALUES (?, ?)", (user, msg))
    conn.commit()
    conn.close()

def get_memory(limit=30):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT user, msg FROM memory ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))

init_db()

async def generate_reply(user, message):
    add_message(user, message)

    history = get_memory(30)
    history_text = "\n".join([f"{u}: {m}" for u, m in history])
    prompt = f"The following is a Discord server chat. Respond naturally and funny like one of them.\n\n{history_text}\nBot:"

    try:
        response = client_hf.text_generation(prompt, max_new_tokens=150, temperature=0.8)
        return response.strip()
    except Exception as e:
        return f"⚠️ HF error: {e}"

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if bot.user.mentioned_in(message) or message.content.startswith("!"):
        reply = await generate_reply(str(message.author), message.content)
        await message.channel.send(reply)

bot.run(DISCORD_TOKEN)
