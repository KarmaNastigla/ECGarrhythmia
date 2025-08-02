# Импорт необходимых библиотек
from fastapi import FastAPI, UploadFile, File                 # FastAPI для создания API
from fastapi.middleware.cors import CORSMiddleware            # Для обработки CORS
import numpy as np                                            # Для работы с массивами
import pandas as pd                                           # Для обработки CSV файлов
import io                                                     # Для работы с потоками данных
from .model_utils import load_model, preprocess_ecg, predict  # Наши вспомогательные функции

# Создание экземпляра FastAPI приложения с названием
app = FastAPI(title="ECG Arrhythmia Classifier")

# Настройка CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем запросы со всех источников
    allow_methods=["*"],  # Разрешаем все HTTP методы
    allow_headers=["*"],  # Разрешаем все заголовки
)

# Загрузка модели при старте сервера
model = load_model("model/best_model.pth")
'''
Классы классификации аритмий:
    1. N - нормальный ритм
    2. L - блокада левой ножки пучка Гиса
    3. R - блокада правой ножки пучка Гиса
    4. A - предсердная экстрасистолия
    5. V - желудочковая экстрасистолия
'''
classes = ['N', 'L', 'R', 'A', 'V']


@app.post("/predict")
async def predict_ecg(file: UploadFile = File(...)):
    """
    Эндпоинт для классификации ЭКГ сигнала из CSV файла.
    Принимает файл и возвращает предсказанный класс и вероятности для всех классов.

    Args:
        file (UploadFile): Загруженный CSV файл с ЭКГ данными

    Returns:
        dict: Словарь с предсказанным классом и вероятностями, для всех классов

    """
    # 1. Чтение CSV файла из запроса
    content = await file.read()                # Асинхронное чтение содержимого файла

    # 2. Преобразование содержимого в DataFrame через StringIO (создание файлоподобный объект из строки)
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))

    # 3. Получаем сигнал ЭКГ, обрабатывая разные варианты названий колонок
    mlii_col = "'MLII'" if "'MLII'" in df.columns else "MLII"
    signal = df[mlii_col].values                # Получаем numpy массив значений

    # 4. Предобработка сигнала и предсказание
    input_tensor = preprocess_ecg(signal)        # Преобразуем сигнал в формат для модели
    probabilities = predict(model, input_tensor) # Получаем вероятности классов

    # Определяем класс с максимальной вероятностью
    predicted_class = classes[np.argmax(probabilities)]

    # Формируем ответ в формате JSON
    return {
        "class": predicted_class,                # Предсказанный класс
        "probabilities": {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    }


@app.get("/health")
def health_check():
    """
    Эндпоинт для проверки работоспособности сервиса.
    Используется для мониторинга и проверки доступности API.

    Returns:
        dict: Статус сервиса (всегда {"status": "OK"})

    """
    return {"status": "OK"}