import os
import discord
import requests

# Environment variables (set these in Railway/your host)
TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # you can change to another HF model

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# store last messages per user
user_histories = {}

def query_huggingface(prompt: str, user_id: str) -> str:
    history = user_histories.get(user_id, [])
    context = ""
    if history:
        # include last few messages in context
        for msg in history[-10:]:
            context += msg + "\n"
        context += "\n"  # separator

    full_prompt = context + prompt

    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL}",
        headers=headers,
        json={"inputs": full_prompt, "parameters": {"max_new_tokens": 150}}
    )
    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        return f"⚠️ HF API error: {data['error']}"
    # data[0]["generated_text"] is typical shape
    return data[0].get("generated_text", "")

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")

@client.event
async def on_message(message):
    # ignore bot messages
    if message.author.bot:
        return

    user_id = str(message.author.id)
    content = message.content.strip()

    # Save user message to history
    if user_id not in user_histories:
        user_histories[user_id] = []
    user_histories[user_id].append(content)

    # Only respond when bot is mentioned
    if client.user in message.mentions:
        # remove mention text
        cleaned = content.replace(f"<@{client.user.id}>", "").strip()
        if not cleaned:
            await message.channel.send("You mentioned me but said nothing 🤔")
            return

        await message.channel.typing()
        reply = query_huggingface(cleaned, user_id)
        if not reply:
            reply = "(no answer)"

        # split in chunks (Discord limit ~2000 chars)
        for chunk in [reply[i:i+1900] for i in range(0, len(reply), 1900)]:
            await message.channel.send(chunk)

if __name__ == "__main__":
    if not TOKEN or not HF_TOKEN:
        print("Missing DISCORD_TOKEN or HF_TOKEN environment variable!")
    else:
        client.run(TOKEN)
