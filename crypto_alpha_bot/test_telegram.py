import asyncio
import aiohttp
import yaml
import os
import sys

# Caminho absoluto da raiz (pasta onde este arquivo está)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construir caminho absoluto para secrets.yaml
SECRETS_PATH = os.path.join(BASE_DIR, "config", "secrets.yaml")

print("DEBUG: BASE_DIR =", BASE_DIR)
print("DEBUG: SECRETS_PATH =", SECRETS_PATH)
print("DEBUG: Working dir (os.getcwd()) =", os.getcwd())
print("DEBUG: Exists(secrets)?", os.path.exists(SECRETS_PATH))

if not os.path.exists(SECRETS_PATH):
    print("ERRO: secrets.yaml NÃO encontrado em", SECRETS_PATH)
    # Listar conteúdo da pasta config para depuração
    cfg_dir = os.path.join(BASE_DIR, "config")
    if os.path.isdir(cfg_dir):
        print("Conteúdo de config/:", os.listdir(cfg_dir))
    else:
        print("Pasta config/ não existe em", cfg_dir)
    sys.exit(1)

# Ler secrets
try:
    with open(SECRETS_PATH, "r", encoding="utf-8") as fh:
        secrets = yaml.safe_load(fh) or {}
except Exception as e:
    print("ERRO ao ler YAML:", e)
    sys.exit(1)

tg = secrets.get("telegram", {})
TOKEN = tg.get("bot_token")
CHAT_ID = tg.get("chat_id")

print("DEBUG: TOKEN presente?", bool(TOKEN))
print("DEBUG: CHAT_ID =", CHAT_ID)

if not TOKEN or not CHAT_ID:
    print("ERRO: bot_token ou chat_id ausente em secrets.yaml")
    sys.exit(1)


async def send_test():
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": "Teste cry2muchbot ✅ (absolute path)"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=15) as r:
            status = r.status
            text = await r.text()
            print("HTTP Status:", status)
            print("Resposta:", text)


if __name__ == "__main__":
    asyncio.run(send_test())
