import json
import requests
import time

# Загрузка тестовых данных
with open('./data/test_data.json', 'r') as f:
    test_data = json.load(f)
    print(test_data)

url = "http://127.0.0.1:8000/predict"

start_time = time.perf_counter()

# Отправка POST-запросов
for data_element in test_data:
    response = requests.post(url, json=data_element)

    # Проверка и вывод результата
    if response.status_code == 200:
        print("Тест пройден успешно!")
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)

end_time = time.perf_counter()
print(f"Время выполнения: {end_time - start_time:.2f} секунд")

