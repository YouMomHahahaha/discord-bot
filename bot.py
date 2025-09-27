import os
import discord
import requests

TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Store per-user chat history
user_histories = {}

def query_huggingface(prompt, user_id):
    # Build conversation history
    history = user_histories.get(user_id, [])
    context = ""
    if history:
        context = "Here is this user's chat history:\n"
        for msg in history[-20:]:  # only last 20 messages
            context += f"- {msg}\n"
        context += "\nReply naturally, in their style.\n\n"

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL}",
        headers=headers,
        json={"inputs": context + prompt, "parameters": {"max_new_tokens": 200}}
    )
    data = response.json()
    if isinstance(data, dict) and "error" in data:
        return f"⚠️ HF API Error: {data['error']}"
    return data[0]["generated_text"]

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    user_id = str(message.author.id)

    # Save every user message to history
    if user_id not in user_histories:
        user_histories[user_id] = []
    user_histories[user_id].append(message.content)

    # Only reply if bot is mentioned
    if client.user in message.mentions:
        user_input = message.content.replace(f"<@{client.user.id}>", "").strip()
        if not user_input:
            await message.channel.send("You mentioned me, but didn’t say anything 🤔")
            return

        await message.channel.typing()
        reply = query_huggingface(user_input, user_id)

        # Discord message size limit fix
        for chunk in [reply[i:i+1900] for i in range(0, len(reply), 1900)]:
            await message.channel.send(chunk)

if __name__ == "__main__":
    client.run(TOKEN)
