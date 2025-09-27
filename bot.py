import os
import discord
import requests
from discord.ext import commands

# Load tokens from environment
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not DISCORD_TOKEN:
    raise ValueError("❌ Missing DISCORD_TOKEN environment variable!")
if not HF_TOKEN:
    raise ValueError("❌ Missing HF_TOKEN environment variable!")

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/gpt2"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Create bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def query_huggingface(prompt: str) -> str:
    payload = {"inputs": prompt}

    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)

        # Check if API returned success
        if resp.status_code != 200:
            return f"⚠️ HF API error {resp.status_code}: {resp.text}"

        # Try to decode JSON safely
        try:
            data = resp.json()
        except Exception as e:
            return f"⚠️ JSON decode failed: {e}\nResponse: {resp.text[:200]}"

        # Hugging Face text response
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            return f"⚠️ Unexpected HF response: {data}"

    except Exception as e:
        return f"⚠️ Request failed: {e}"

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content.startswith("!ask"):
        prompt = message.content[len("!ask "):].strip()
        if not prompt:
            await message.channel.send("Please provide a question after `!ask`.")
            return

        reply = query_huggingface(prompt)
        await message.channel.send(reply)

# Run bot
bot.run(DISCORD_TOKEN)
