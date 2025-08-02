import torch
from torch import nn
import numpy as np
import pywt             # Для вейвлет-преобразований (используется для денойзинга)
from scipy import stats # Для статистических операций (z-score нормализация)


class ECGCNN(nn.Module):
    """
    Архитектура CNN для классификации сигналов ЭКГ.

    Особенности:
    - Последовательность сверточных блоков с постепенным увеличением каналов
    - BatchNorm и Dropout для регуляризации
    - AdaptiveAvgPool для приведения к фиксированному размеру
    - Двухслойный классификатор с dropout

    """

    def __init__(self):
        """Инициализация слоев модели"""
        super().__init__()              # Инициализация родительского класса nn.Module

        # Сверточные слои для извлечения признаков
        self.features = nn.Sequential(
            # Первый сверточный блок: 32 канала, ядро 15, padding 7 для сохранения размера
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),         # Нормализация по batch для стабилизации обучения
            nn.ReLU(),                  # Функция активации
            nn.MaxPool1d(3, stride=2),  # Уменьшение размерности в 2 раза

            # Второй сверточный блок: 64 канала, ядро 17
            nn.Conv1d(32, 64, kernel_size=17, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),             # Регуляризация - отключает 30% нейронов
            nn.MaxPool1d(3, stride=2),

            # Третий сверточный блок: 128 каналов, ядро 19
            nn.Conv1d(64, 128, kernel_size=19, padding=9),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(3, stride=2),

            # Адаптивный пулинг к фиксированному размеру (1 точка на канал)
            nn.AdaptiveAvgPool1d(1)  # Преобразует любой размер входа к 1 значению на канал
        )

        # Полносвязные слои для классификации
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),     # Больший dropout для регуляризации FC слоев
            nn.Linear(128, 64),  # Уменьшение размерности признаков
            nn.ReLU(),
            nn.Linear(64, 5)      # Выходной слой (5 классов)
        )

    def forward(self, x):
        """
        Прямой проход модели

        Параметр: x (torch.Tensor): Входной тензор формы [batch_size, 1, signal_length]

        Возвращает: torch.Tensor: Выходной тензор формы [batch_size, num_classes]

        """
        # 1. Извлечение признаков
        x = self.features(x)  # Применяем последовательность сверточных слоев

        # 2. Преобразование формы для классификатора
        x = x.view(x.size(0), -1)  # Разворачиваем в [batch_size, features]

        # 3. Классификация
        return self.classifier(x)


def load_model(model_path):
    """
    Загрузка предобученной модели из файла

        Args:
            model_path: Путь к файлу с весами модели
        Returns:
            Загруженная модель в режиме eval()

    """
    # Определяем устройство (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Создаем экземпляр модели
    model = ECGCNN()

    # Загружаем веса
    checkpoint = torch.load(model_path, map_location=device)

    # Загружаем только веса (игнорируя другие параметры чекпоинта)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model


def preprocess_ecg(signal):
    """
    Предобработка сигнала ЭКГ перед подачей в модель:
    1. Денойзинг с помощью вейвлет-преобразования
    2. Нормализация (z-score)
    3. Приведение к фиксированной длине (360 отсчетов)

    Args:
        signal: Исходный сигнал ЭКГ (1D массив)
    Returns:
        Тензор [1, 1, 360], готовый для подачи в модель

    """
    # 1. Денойзинг с помощью вейвлет-преобразования
    coeffs = pywt.wavedec(signal, 'sym4')
    threshold = 0.04        # Порог для удаления шума
    # Пороговая обработка коэффициентов
    denoised = pywt.waverec(
        [pywt.threshold(c, threshold * max(c)) for c in coeffs],
        'sym4'
    )

    # 2. Нормализация (z-score)
    normalized = stats.zscore(denoised)

    # 3. Приведение к фиксированной длине 360
    if len(normalized) < 360:
        # Дополняем нулями если сигнал короче
        padded = np.pad(normalized, (0, 360 - len(normalized)))
    else:
        # Обрезаем если сигнал длиннее
        padded = normalized[:360]

    # 4. Преобразуем в тензор с правильной размерностью [batch=1, channels=1, length=360]
    return torch.FloatTensor(padded).unsqueeze(0).unsqueeze(0)  # [1, 1, 360]


def predict(model, input_tensor):
    """
    Выполнение предсказания на предобработанном сигнале
    Args:
        model: Загруженная модель
        input_tensor: Тензор с предобработанным сигналом
    Returns:
        Массив вероятностей для каждого класса

    """
    with torch.no_grad():             # Отключаем вычисление градиентов
        output = model(input_tensor)  # Прямой проход
        # Преобразуем выходы в вероятности с помощью softmax
        probabilities = torch.softmax(output, dim=1)

    # Конвертируем в numpy массив и выравниваем
    return probabilities.numpy().flatten()