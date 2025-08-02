# Импорт необходимых библиотек
import streamlit as st           # Для создания веб-интерфейса
import requests                  # Для HTTP-запросов к бэкенду
import pandas as pd              # Для обработки CSV данных
import matplotlib.pyplot as plt  # Для визуализации ЭКГ сигнала
import io                        # Для работы с потоковыми данными

# Настройка страницы Streamlit
st.set_page_config(
    page_title="ECG Classifier",          # Заголовок страницы
    layout="wide"                         # Широкий макет для лучшего отображения
)
st.title("Классификатор аритмий по сигналу ЭКГ")  # Основной заголовок приложения

# URL API бэкенда (используем имя сервиса из docker-compose)
BACKEND_URL = "http://backend:8000/predict"

# Виджет для загрузки файла
uploaded_file = st.file_uploader(
    "Загрузите файл с ЭКГ сигналом (CSV)",  # Текст подсказки
    type="csv"                              # Разрешаем только CSV файлы
)

# Если файл загружен
if uploaded_file is not None:
    try:
        # Чтение и декодирование содержимого файла
        content = uploaded_file.read().decode('utf-8')
        # Преобразование в DataFrame через StringIO (имитация файла)
        df = pd.read_csv(io.StringIO(content))

        # Проверка наличия обязательных колонок
        if "'MLII'" not in df.columns or "'V1'" not in df.columns:
            st.error("Файл должен содержать колонки 'MLII' и 'V1'")
            st.stop()       # Останавливаем выполнение если колонок нет

        # Извлекаем сигнал ЭКГ (обрабатываем оба варианта названий колонок)
        ecg_signal = df["'MLII'"].values if "'MLII'" in df.columns else df["MLII"].values

        # Визуализация ЭКГ (первые 1000 точек для наглядности)
        st.subheader("Визуализация ЭКГ сигнала")
        fig, ax = plt.subplots(figsize=(12, 4))     # Создаем график
        ax.plot(ecg_signal[:1000])                  # Показываем первые 1000 точек
        ax.set_xlabel("Время (отсчеты)")            # Подпись оси X
        ax.set_ylabel("Амплитуда")                  # Подпись оси Y
        st.pyplot(fig)                              # Отображаем график в Streamlit

        # Кнопка для запуска классификации
        if st.button("Классифицировать", type="primary"):
            with st.spinner("Анализ ЭКГ..."):       # Показываем индикатор загрузки
                # Подготавливаем файл для отправки
                files = {
                    "file": (
                        uploaded_file.name,  # Имя файла
                        content,             # Содержимое файла
                        "text/csv"           # MIME-тип
                    )
                }

                try:
                    # Отправляем POST-запрос к бэкенду
                    response = requests.post(BACKEND_URL, files=files)
                    response.raise_for_status()   # Проверяем на ошибки

                    # Парсим JSON ответ
                    result = response.json()

                    # Выводим результаты классификации
                    st.subheader("Результаты классификации")

                    # Словарь с описаниями классов аритмий
                    class_descriptions = {
                        "N": "Нормальный ритм",
                        "L": "Блокада левой ножки пучка Гиса",
                        "R": "Блокада правой ножки пучка Гиса",
                        "A": "Предсердная экстрасистолия",
                        "V": "Желудочковая экстрасистолия"
                    }

                    # Получаем предсказанный класс (по умолчанию "N")
                    predicted_class = result.get("class", "N")

                    # Выводим основной диагноз с цветовым оформлением
                    st.success(f"**Основной диагноз:** {class_descriptions.get(predicted_class, predicted_class)}")

                    # Визуализация вероятностей по классам
                    st.subheader("Вероятности классов")

                    if "probabilities" in result:
                        probs = result["probabilities"]
                        cols = st.columns(5)           # Создаем 5 колонок

                        # Выводим метрики для каждого класса
                        for i, (cls, prob) in enumerate(probs.items()):
                            with cols[i]:
                                # Определяем цвет в зависимости от вероятности
                                color = "red" if prob > 0.7 else "orange" if prob > 0.3 else "green"
                                st.metric(
                                    label=class_descriptions.get(cls, cls),
                                    value=f"{prob:.1%}",
                                    delta="высокая вероятность" if prob > 0.7 else None
                                )

                    # Дополнительная информация (разворачиваемый блок)
                    with st.expander("Подробная информация"):
                        st.json(result)        # Выводим сырой JSON ответ

                # Обработка ошибок соединения
                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка соединения с сервером: {str(e)}")
                # Обработка ошибок парсинга JSON
                except ValueError as e:
                    st.error(f"Ошибка обработки ответа: {str(e)}")

    # Обработка ошибок при чтении файла
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")