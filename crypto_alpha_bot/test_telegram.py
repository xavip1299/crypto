import asyncio, aiohttp, yaml, os, sys

SECRETS_PATH = "config/secrets.yaml"

if not os.path.exists(SECRETS_PATH):
    print(f"Arquivo {SECRETS_PATH} não encontrado. Crie-o antes de testar.")
    sys.exit(1)

secrets = yaml.safe_load(open(SECRETS_PATH))
tg = secrets.get("telegram", {})
TOKEN = tg.get("bot_token")
CHAT_ID = tg.get("chat_id")

if not TOKEN or not CHAT_ID:
    print("Token ou chat_id ausentes em config/secrets.yaml.")
    sys.exit(1)

async def main():
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": "Teste cry2muchbot ✅"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as r:
            print("Status:", r.status)
            print("Resposta:", await r.text())

asyncio.run(main())
