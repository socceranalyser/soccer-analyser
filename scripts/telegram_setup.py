r"""Connect the Telegram bot.

1. In Telegram open @BotFather -> /newbot -> copy the token.
2. Put it into the file .env in the project folder:   TELEGRAM_BOT_TOKEN=123456:ABC...
3. Open your new bot in Telegram and press Start (send /start).
4. Run:  .venv\Scripts\python.exe scripts\telegram_setup.py
   -> finds your chat id, saves it to .env and sends a test message.
"""
import requests

from soccer.notify import load_config, save_env_value, send

if __name__ == "__main__":
    token = load_config().get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Нет TELEGRAM_BOT_TOKEN в файле .env — см. инструкцию в начале файла.")
    upd = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30).json()
    if not upd.get("ok"):
        raise SystemExit(f"Telegram отклонил токен: {upd.get('description')}")
    chats = [u["message"]["chat"] for u in upd.get("result", []) if "message" in u]
    if not chats:
        raise SystemExit("Бот ещё не получил сообщений: откройте бота в Telegram, нажмите Start "
                         "и запустите скрипт снова.")
    chat = chats[-1]
    save_env_value("TELEGRAM_CHAT_ID", str(chat["id"]))
    ok = send("✅ Soccer Analyser подключён. Ежедневные прогнозы будут приходить сюда.")
    print(f"chat id сохранён ({chat.get('first_name', '')}); тестовое сообщение: "
          f"{'отправлено' if ok else 'НЕ отправлено'}")
