import os
import discord
from discord.ext import commands
import requests

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "microsoft/Phi-3-mini-4k-instruct"  # pick your model

intents = discord.Intents.default()
intents.message_content = True  # enable reading message text
bot = commands.Bot(command_prefix="!", intents=intents)

def query_huggingface(prompt: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL}",
        headers=headers,
        json={"inputs": prompt},
    )
    if resp.status_code != 200:
        print("HF error:", resp.text)
        return "⚠️ HF API error"
    try:
        data = resp.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return str(data)
    except Exception as e:
        print("Decode error:", e, resp.text)
        return "⚠️ Couldn’t parse HF response"

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Bot only replies when pinged
    if bot.user.mentioned_in(message):
        cleaned = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if not cleaned:
            return
        reply = query_huggingface(cleaned)
        await message.reply(reply)

    # Important: let commands still work
    await bot.process_commands(message)

bot.run(DISCORD_TOKEN)
