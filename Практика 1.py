import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Создание набора данных для обучения
np.random.seed(42)
num_samples = 10000

# Генерация случайных пар чисел и их среднего арифметического
X = np.random.uniform(-100, 100, size=(num_samples, 2))
y = (X[:, 0] + X[:, 1]) / 2  # Среднее арифметическое

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных (не обязательно для такой простой задачи, но полезно для демонстрации)
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

# Создание модели нейронной сети
model = keras.Sequential([
    layers.Input(shape=(2,)),  # Два входных нейрона для двух чисел
    layers.Dense(16, activation='relu', name='hidden1'),
    layers.Dense(8, activation='relu', name='hidden2'),
    layers.Dense(1, name='output')  # Один выходной нейрон
])

# Компиляция модели
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',  # Среднеквадратичная ошибка
    metrics=['mae']  # Средняя абсолютная ошибка
)

# Вывод структуры модели
print("Структура нейронной сети:")
model.summary()

# Обучение модели
print("\nОбучение модели...")
history = model.fit(
    X_train_norm, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

# Оценка модели на тестовых данных
print("\nОценка на тестовых данных...")
test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"Средняя абсолютная ошибка на тестовых данных: {test_mae:.6f}")

# Визуализация процесса обучения
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Ошибка обучения')
plt.plot(history.history['val_loss'], label='Ошибка валидации')
plt.title('Динамика ошибки')
plt.xlabel('Эпоха')
plt.ylabel('Среднеквадратичная ошибка')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='MAE обучения')
plt.plot(history.history['val_mae'], label='MAE валидации')
plt.title('Динамика средней абсолютной ошибки')
plt.xlabel('Эпоха')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Проверка модели на нескольких примерах
print("\nПроверка на примерах:")
test_examples = np.array([
    [10, 20],    # Среднее: 15
    [-5, 5],     # Среднее: 0
    [100, 200],  # Среднее: 150
    [7.5, 2.5],  # Среднее: 5
    [0, 0]       # Среднее: 0
])

test_examples_norm = (test_examples - X_mean) / X_std
predictions = model.predict(test_examples_norm, verbose=0)

for i, example in enumerate(test_examples):
    true_value = (example[0] + example[1]) / 2
    predicted_value = predictions[i][0]
    error = abs(true_value - predicted_value)
    print(f"Числа: {example[0]:.2f} и {example[1]:.2f}")
    print(f"  Истинное среднее: {true_value:.6f}")
    print(f"  Предсказанное среднее: {predicted_value:.6f}")
    print(f"  Ошибка: {error:.6f}")
    print()

# Создание упрощенной модели для демонстрации работы нейронной сети
print("\n" + "="*60)
print("Упрощенная модель для лучшего понимания работы сети")
print("="*60)

# Создаем простую модель с понятными весами
simple_model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(1, use_bias=False)  # Просто линейная комбинация входов
])

# Устанавливаем веса вручную для вычисления среднего арифметического
# Для среднего арифметического: выход = 0.5 * x1 + 0.5 * x2
weights = np.array([[0.5], [0.5]])
simple_model.layers[0].set_weights([weights])

# Проверка упрощенной модели
print("\nПроверка упрощенной модели:")
for example in test_examples:
    input_example = example.reshape(1, 2)
    prediction = simple_model.predict(input_example, verbose=0)
    true_value = (example[0] + example[1]) / 2
    print(f"Числа: {example[0]:.2f} и {example[1]:.2f}")
    print(f"  Результат: {prediction[0][0]:.6f} (ожидалось: {true_value:.6f})")
    print()

# Анализ обученной модели
print("\n" + "="*60)
print("Анализ обученной модели")
print("="*60)

# Получаем веса из первого скрытого слоя
weights_hidden1 = model.get_layer('hidden1').get_weights()[0]
biases_hidden1 = model.get_layer('hidden1').get_weights()[1]

print(f"Размерность весов первого скрытого слоя: {weights_hidden1.shape}")
print(f"Количество нейронов в первом скрытом слое: {weights_hidden1.shape[1]}")
print(f"\nНейронная сеть успешно обучена вычислению среднего арифметического!")
print(f"Средняя абсолютная ошибка: {test_mae:.6f}")