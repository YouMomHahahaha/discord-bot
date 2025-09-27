import os
import discord
import sqlite3
from discord.ext import commands
from huggingface_hub import InferenceClient

# === CONFIG ===
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")   # put this in Railway secrets
HF_TOKEN = os.getenv("HF_TOKEN")             # Hugging Face token
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# === DATABASE ===
DB_FILE = "chat_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            user_id TEXT,
            username TEXT,
            message TEXT,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_memory(user_id, username, message, response):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO memory VALUES (?, ?, ?, ?)", (user_id, username, message, response))
    conn.commit()
    conn.close()

def load_memory(user_id, limit=5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT message, response FROM memory WHERE user_id=? ORDER BY rowid DESC LIMIT ?", (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]  # return oldest first

# === HF CLIENT ===
hf_client = InferenceClient(model=MODEL, token=HF_TOKEN)

async def query_hf(user_prompt, history=[]):
    try:
        # build chat-style messages
        messages = []
        for (u, b) in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": b})

        messages.append({"role": "user", "content": user_prompt})

        response = hf_client.chat_completion(
            model=MODEL,
            messages=messages,
            max_tokens=256
        )

        return response.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ HF API error: {str(e)}"

# === DISCORD BOT ===
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    init_db()

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if bot.user.mentioned_in(message):
        user_id = str(message.author.id)
        username = str(message.author.name)
        prompt = message.content.replace(f"<@{bot.user.id}>", "").strip()

        history = load_memory(user_id)
        reply = await query_hf(prompt, history)

        await message.channel.send(reply)

        save_memory(user_id, username, prompt, reply)

    await bot.process_commands(message)

# === RUN ===
bot.run(DISCORD_TOKEN)
