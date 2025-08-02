# ECG Arrhythmia Classification Project

## 🌟 Основные возможности

| Особенность      | Описание                                |
|------------|-----------------------------------------|
| 🧠 Умная классификация | Нейросетевая модель с точностью 99.4%   |
| 📈 Профессиональная визуализация | Интерактивные графики ЭКГ с аннотациями |
| ⚡ Реальное время   | Мгновенный анализ при загрузке данных   |
| 🏥 5 типов аритмий | Полный спектр распространенных нарушений | 
	
🏥 Классы аритмий:
- N - Нормальный ритм
- L - Блокада левой ножки
- R - Блокада правой ножки
- A - Предсердная экстрасистолия
- V - Желудочковая экстрасистолия

## 🛠 Технологический стек
Backend
<p> <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python" alt="Python"> <img src="https://img.shields.io/badge/FastAPI-0.95-green?logo=fastapi" alt="FastAPI"> <img src="https://img.shields.io/badge/PyTorch-1.13-red?logo=pytorch" alt="PyTorch"> </p>
Frontend
<p> <img src="https://img.shields.io/badge/Streamlit-1.18-ff4b4b?logo=streamlit" alt="Streamlit"> <img src="https://img.shields.io/badge/Altair-4.2-yellow?logo=vega" alt="Altair"> </p>
Инфраструктура
<p> <img src="https://img.shields.io/badge/Docker-20.10+-2496ED?logo=docker" alt="Docker"> <img src="https://img.shields.io/badge/Compose-2.0+-384d54?logo=docker" alt="Docker Compose"> </p>

## 🚀 Быстрый старт
``` bash
# Клонируем репозиторий с GitHub
git clone https://github.com/KarmaNastigla/ECGarrhythmia

# Переходим в директорию проекта 
cd ECGarrhythmia

# Собираем и запускаем контейнеры
docker-compose up --build

# Чтобы остановить контейнеры выполните:
# docker-compose down 
```

После запуска откройте в браузере:

- 🖥 Интерфейс: http://localhost:8501
- 📚 API Docs: http://localhost:8000/docs

## 💡 Как использовать
1. Загрузите CSV файл с ЭКГ данными
2. Просмотрите визуализацию сигнала
3. Получите диагноз с вероятностями

Ссылка, где можно взять csv фыйлы с данными по ЭКГ:

https://www.kaggle.com/code/karanastigla/mit-bih-arrhythmia-pytorch-99-4/input

Пример данных:
``` bash
time, lead_I, lead_II
0.000, -0.145, 0.212
0.004, -0.142, 0.210
...
```