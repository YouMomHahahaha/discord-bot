import os
import json
import discord
from discord.ext import commands
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

client_hf = InferenceClient(model=MODEL, token=HF_TOKEN)

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

DB_FILE = "chat_memory.json"

# Load DB
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r", encoding="utf-8") as f:
        chat_memory = json.load(f)
else:
    chat_memory = []

def save_memory():
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_memory, f, ensure_ascii=False, indent=2)

def build_prompt(message):
    # Take last 30 messages as "training context"
    context = "\n".join([f"{m['username']}: {m['content']}" for m in chat_memory[-30:]])
    return f"""
This is a Discord server chat. People are friends, make jokes, and talk casually.
The bot must act like one of them, matching their style and humor.

Server conversation so far:
{context}

{message.author.display_name}: {message.content}
Bot:"""

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Log every message into DB
    chat_memory.append({"username": message.author.display_name, "content": message.content})
    save_memory()

    # Only reply if bot is mentioned
    if bot.user.mentioned_in(message):
        prompt = build_prompt(message)

        try:
            response = client_hf.text_generation(
                prompt,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True
            )
            reply = response.strip()

            # Save bot’s reply too
            chat_memory.append({"username": "Bot", "content": reply})
            save_memory()

            await message.reply(reply)

        except Exception as e:
            await message.reply(f"⚠️ HF API error: {e}")

bot.run(DISCORD_TOKEN)
