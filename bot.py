import os
import discord
import sqlite3
import requests
from discord.ext import commands
from huggingface_hub import InferenceClient

# === CONFIG ===
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")   # railway secret
HF_TOKEN = os.getenv("HF_TOKEN")             # hugging face token
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
    return rows[::-1]  # oldest first

# === SIMPLE WEB SEARCH ===
def web_search(query, max_results=3):
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "t": "discord-bot"}
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        results = []
        for r in data.get("RelatedTopics", []):
            if isinstance(r, dict):
                txt = r.get("Text")
                if txt:
                    results.append(txt)
            if len(results) >= max_results:
                break
        return results
    except Exception:
        return []

# === HF CLIENT ===
hf_client = InferenceClient(model=MODEL, token=HF_TOKEN)

async def query_hf(user_prompt, history=[], username="user"):
    try:
        # grab some context from web
        web_info = web_search(user_prompt)
        context = "\n".join(web_info) if web_info else "no useful info found online."

        # bot style: lowercase, bored
        messages = [
            {"role": "system", "content": "you are a discord bot that always replies in lowercase, short, and bored. no uppercase, emojis, or excitement. reply like a normal user, not like a bot."},
            {"role": "system", "content": f"web context (might help): {context}"}
        ]

        # include memory
        for (u, b) in history:
            messages.append({"role": "user", "content": f"{username}: {u}"})
            messages.append({"role": "assistant", "content": b})

        messages.append({"role": "user", "content": f"{username}: {user_prompt}"})

        response = hf_client.chat_completion(
            model=MODEL,
            messages=messages,
            max_tokens=128
        )

        return response.choices[0].message["content"].lower().strip()
    except Exception as e:
        return f"⚠️ hf api error: {str(e)}"

# === DISCORD BOT ===
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ logged in as {bot.user}")
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
        reply = await query_hf(prompt, history, username)

        await message.channel.send(reply)

        save_memory(user_id, username, prompt, reply)

    await bot.process_commands(message)

# === RUN ===
bot.run(DISCORD_TOKEN)
