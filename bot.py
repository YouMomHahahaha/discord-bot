import discord
from discord.ext import commands
import sqlite3
from huggingface_hub import InferenceClient
import os

# ---- CONFIG ----
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")   # put in railway vars
HF_TOKEN = os.getenv("HF_TOKEN")             # huggingface token
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# ---- HF CLIENT ----
hf_client = InferenceClient(HF_TOKEN)

# ---- DISCORD BOT ----
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ---- DATABASE ----
DB_FILE = "memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS memory (
        user_id TEXT,
        username TEXT,
        user_msg TEXT,
        bot_reply TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

def load_memory(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT user_msg, bot_reply FROM memory WHERE user_id=? ORDER BY rowid DESC LIMIT 5", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]  # oldest → newest

def save_memory(user_id, username, user_msg, bot_reply):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO memory (user_id, username, user_msg, bot_reply) VALUES (?, ?, ?, ?)",
              (user_id, username, user_msg, bot_reply))
    conn.commit()
    conn.close()

# ---- HF QUERY ----
async def query_hf(user_prompt, history=[], username="user"):
    try:
        messages = [
            {"role": "system", "content": f"you are a discord bot that always replies in lowercase, short, and bored. you know the user is named {username}. sometimes say their name, but stay uninterested."}
        ]

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

# ---- DISCORD EVENTS ----
@bot.event
async def on_ready():
    print(f"logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
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

# ---- RUN ----
bot.run(DISCORD_TOKEN)
