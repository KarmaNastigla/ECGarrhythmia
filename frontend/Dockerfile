# Базовый образ - официальный Python 3.9 на slim-версии Debian
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем ТОЛЬКО файл зависимостей в контейнер
# Делаем это отдельным шагом для эффективного использования кэша Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем основной файл приложения
COPY ./app.py .

# Команда запуска приложения при старте контейнера:
# streamlit run - запуск Streamlit-приложения
# app.py - основной файл приложения
# --server.port=8501 - фиксируем порт Streamlit (по умолчанию 8501)
# --server.address=0.0.0.0 - слушаем все сетевые интерфейсы (важно для Docker)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]