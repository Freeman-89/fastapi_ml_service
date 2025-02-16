# ML Prediction API

## Описание

Проект представляет собой API-сервис на FastAPI для предсказания пользовательского поведения на основе входных данных с параметрами рекламы, устройства и геолокации.

## Структура проекта

- **main.py** — код FastAPI приложения
- **model/** — директория с сохранённой моделью
- **data/** — входные тестовые данные
- **test.py** — скрипт интеграционного тестирования
- **pipeline.py** — скрипт для построения и обучения модели

## Запуск

1. **Клонировать репозиторий:**

   ```bash
   git clone https://github.com/your_username/ml_predictor_api.git
   cd ml_predictor_api
   ```

2. **Создать виртуальное окружение и установить зависимости:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # для Linux
   .venv\Scripts\activate     # для Windows
   pip install -r requirements.txt
   ```

3. **Запустить приложение:**

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Использование

### Проверка состояния

```bash
curl -X GET http://127.0.0.1:8000/status
```

### Получение версии модели

```bash
curl -X GET http://127.0.0.1:8000/version
```

### Предсказание

Пример запроса:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "utm_source": "google",
    "utm_medium": "cpc",
    "utm_campaign": "summer_sale",
    "utm_adcontent": "discount_50",
    "utm_keyword": "buy shoes",
    "device_category": "mobile",
    "device_os": "Android",
    "device_brand": "Samsung",
    "device_screen_resolution": "1080x2400",
    "device_browser": "Chrome",
    "geo_country": "Russia",
    "geo_city": "Moscow"
}'
```

### Ожидаемый ответ

```json
{
    "form": {
        "utm_source": "google",
        "utm_medium": "cpc",
        "utm_campaign": "summer_sale",
        "utm_adcontent": "discount_50",
        "utm_keyword": "buy shoes",
        "device_category": "mobile",
        "device_os": "Android",
        "device_brand": "Samsung",
        "device_screen_resolution": "1080x2400",
        "device_browser": "Chrome",
        "geo_country": "Russia",
        "geo_city": "Moscow"
    },
    "Result": 1
}
```

## Тестирование

Для выполнения теста используется скрипт `test.py`.

```bash
python test.py
```

## Измерение времени выполнения

В консоли будет отображено время выполнения запросов к серверу.

## Построение и обучение модели

Для построения и обучения модели запускается скрипт `pipeline.py`.

```bash
python pipeline.py
```

В скрипте реализованы следующие этапы:

- Обработка данных (импьютация, масштабирование, one-hot кодирование);
- Обучение нескольких моделей (Logistic Regression, Random Forest, SVC, MLP);
- Оценка моделей с помощью кросс-валидации по метрике ROC AUC;
- Выбор наилучшей модели и сохранение её с помощью dill.

### Логика работы pipeline

1. Загрузка данных и их анализ.
2. Применение трансформеров для обработки категориальных и числовых признаков.
3. Обучение нескольких моделей с использованием кросс-валидации.
4. Сравнение моделей по метрике ROC AUC.
5. Сохранение лучшей модели в формате pkl с метаданными (дата, автор, точность).

## Требования

- Python 3.10+
- FastAPI
- Pandas
- Requests
- Scikit-learn
- Dill

## Контакты

Автор: Артемий GitHub: [ml\_predictor\_api](https://github.com/your_username/ml_predictor_api)


