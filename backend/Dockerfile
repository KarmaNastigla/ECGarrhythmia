# Базовый образ - официальный Python 3.9 на slim-версии Debian
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .
# Установка зависимостей Python:
# --no-cache-dir - не кэшировать установленные пакеты (уменьшает размер образа)
# -r requirements.txt - установка всех пакетов из файла
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВСЕ файлы из текущей директории (.) в /app контейнера
COPY . .

# Команда запуска приложения при старте контейнера:
# uvicorn - ASGI-сервер для FastAPI
# app.main:app - импорт приложения из app/main.py
# --host 0.0.0.0 - слушать все сетевые интерфейсы
# --port 8000 - порт для входящих соединений
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]