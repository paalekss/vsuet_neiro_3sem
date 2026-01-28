# -*- coding: utf-8 -*-
"""
Практическая работа №2
Вариант 2: Самостоятельное разбиение обучающей выборки на обучающую и валидационную
ФИО: [Ваше ФИО]
Группа: [Ваша группа]
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

# Настройка стилей графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Отключение предупреждений TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 80)
print("ПРАКТИЧЕСКАЯ РАБОТА №2 - ВАРИАНТ 2")
print("=" * 80)
print("САМОСТОЯТЕЛЬНОЕ РАЗБИЕНИЕ ОБУЧАЮЩЕЙ ВЫБОРКИ")
print("НА ОБУЧАЮЩУЮ И ВАЛИДАЦИОННУЮ (10 000 наблюдений)")
print("=" * 80)

# ============================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================================
print("\n" + "="*60)
print("1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ MNIST")
print("="*60)

# Загрузка данных MNIST
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"Загружено данных:")
print(f"  Обучающая выборка: {x_train_full.shape[0]} изображений, размер {x_train_full.shape[1:]} каждый")
print(f"  Тестовая выборка:  {x_test.shape[0]} изображений, размер {x_test.shape[1:]} каждый")

# Визуализация нескольких изображений из исходного набора
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train_full[i], cmap='gray')
    ax.set_title(f"Цифра: {y_train_full[i]}")
    ax.axis('off')
plt.suptitle("Примеры изображений из MNIST (первые 10)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Нормализация данных
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Преобразование в одномерные массивы (28x28 -> 784)
x_train_full = x_train_full.reshape(-1, 784)  # ИСПРАВЛЕНО: было 28*28, исправлено на 784
x_test = x_test.reshape(-1, 784)  # ИСПРАВЛЕНО: было 28*784, исправлено на 784

print(f"\nПосле преобразования:")
print(f"  x_train_full: {x_train_full.shape}")
print(f"  x_test: {x_test.shape}")

# One-hot encoding для меток классов
y_train_full_cat = keras.utils.to_categorical(y_train_full, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"  y_train_full_cat: {y_train_full_cat.shape}")
print(f"  y_test_cat: {y_test_cat.shape}")

# ============================================================================
# 2. САМОСТОЯТЕЛЬНОЕ РАЗБИЕНИЕ НА ОБУЧАЮЩУЮ И ВАЛИДАЦИОННУЮ ВЫБОРКИ
# ============================================================================
print("\n" + "="*60)
print("2. САМОСТОЯТЕЛЬНОЕ РАЗБИЕНИЕ ВЫБОРКИ")
print("="*60)

print("Исходная обучающая выборка содержит 60 000 изображений")
print("Требуется создать валидационную выборку из 10 000 изображений")
print("Оставшиеся 50 000 будут использоваться для обучения")

# Используем train_test_split для случайного разбиения
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full_cat, 
    test_size=10000,  # Ровно 10 000 изображений
    random_state=42,
    stratify=y_train_full  # Сохраняем распределение классов
)

# Также сохраняем метки в обычном формате для анализа
_, _, y_train_labels, y_val_labels = train_test_split(
    x_train_full, y_train_full, 
    test_size=10000,
    random_state=42,
    stratify=y_train_full
)

print(f"\nРезультат разбиения:")
print(f"  Обучающая выборка:  {x_train.shape[0]} изображений")
print(f"  Валидационная выборка: {x_val.shape[0]} изображений")
print(f"  Тестовая выборка:    {x_test.shape[0]} изображений")

# Проверка распределения классов
print("\nРаспределение классов по выборкам:")

train_classes = np.argmax(y_train, axis=1)
val_classes = y_val_labels
test_classes = np.argmax(y_test_cat, axis=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (classes, title, ax) in enumerate([
    (train_classes, f'Обучающая выборка\n({len(train_classes)} изображений)', axes[0]),
    (val_classes, f'Валидационная выборка\n({len(val_classes)} изображений)', axes[1]),
    (test_classes, f'Тестовая выборка\n({len(test_classes)} изображений)', axes[2])
]):
    unique, counts = np.unique(classes, return_counts=True)
    bars = ax.bar(unique, counts, color=plt.cm.tab10(unique))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Класс (цифра)')
    ax.set_ylabel('Количество')
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3)
    
    # Добавление значений на столбцы
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{count}', ha='center', va='bottom', fontsize=9)

plt.suptitle('РАСПРЕДЕЛЕНИЕ КЛАССОВ ПО ВЫБОРКАМ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 3. СОЗДАНИЕ АРХИТЕКТУРЫ НЕЙРОННОЙ СЕТИ
# ============================================================================
print("\n" + "="*60)
print("3. СОЗДАНИЕ АРХИТЕКТУРЫ НЕЙРОННОЙ СЕТИ")
print("="*60)

model = models.Sequential([
    # Входной слой
    layers.Dense(256, activation='relu', input_shape=(784,), name='dense_1'),
    layers.Dropout(0.3, name='dropout_1'),
    
    # Скрытые слои
    layers.Dense(128, activation='relu', name='dense_2'),
    layers.Dropout(0.3, name='dropout_2'),
    
    layers.Dense(64, activation='relu', name='dense_3'),
    layers.Dropout(0.2, name='dropout_3'),
    
    # Выходной слой
    layers.Dense(10, activation='softmax', name='output')
])

# Вывод структуры сети
print("\nСтруктура нейронной сети:")
model.summary()

# ============================================================================
# 4. КОМПИЛЯЦИЯ И ОБУЧЕНИЕ МОДЕЛИ
# ============================================================================
print("\n" + "="*60)
print("4. КОМПИЛЯЦИЯ И ОБУЧЕНИЕ МОДЕЛИ")
print("="*60)

# Компиляция модели
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Параметры обучения:")
print(f"  Оптимизатор: Adam (learning_rate=0.001)")
print(f"  Функция потерь: categorical_crossentropy")
print(f"  Метрика: accuracy")
print(f"  Размер батча: 128")
print(f"  Количество эпох: 30")

# Callback для ранней остановки (предотвращение переобучения)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Callback для сохранения лучшей модели
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model_mnist.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

# Обучение модели
print("\nНачало обучения...")
start_time = time.time()

history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

training_time = time.time() - start_time
print(f"\nОбучение завершено за {training_time:.2f} секунд")

# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ОБУЧЕНИЯ
# ============================================================================
print("\n" + "="*60)
print("5. ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ОБУЧЕНИЯ")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График функции потерь
axes[0].plot(history.history['loss'], label='Обучающая выборка', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Валидационная выборка', linewidth=2)
axes[0].set_title('ФУНКЦИЯ ПОТЕРЬ (LOSS)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# График точности
axes[1].plot(history.history['accuracy'], label='Обучающая выборка', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Валидационная выборка', linewidth=2)
axes[1].set_title('ТОЧНОСТЬ (ACCURACY)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('ПРОЦЕСС ОБУЧЕНИЯ НЕЙРОННОЙ СЕТИ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 6. ОЦЕНКА КАЧЕСТВА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ
# ============================================================================
print("\n" + "="*60)
print("6. ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
print("="*60)

# Оценка на валидационной выборке
val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"Результаты на валидационной выборке (10 000 изображений):")
print(f"  Loss: {val_loss:.4f}")
print(f"  Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Оценка на тестовой выборке
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nРезультаты на тестовой выборке (10 000 изображений):")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Проверка требования к точности (не менее 97%)
if test_accuracy >= 0.97:
    print(f"\n✓ Требование выполнено: точность ≥ 97%")
else:
    print(f"\n⚠ Требование не выполнено: точность < 97%")

# ============================================================================
# 7. АНАЛИЗ РЕЗУЛЬТАТОВ КЛАССИФИКАЦИИ
# ============================================================================
print("\n" + "="*60)
print("7. АНАЛИЗ РЕЗУЛЬТАТОВ КЛАССИФИКАЦИИ")
print("="*60)

# Предсказания на тестовой выборке
y_pred_proba = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)

# Визуализация матрицы ошибок
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('МАТРИЦА ОШИБОК (CONFUSION MATRIX) - ТЕСТОВАЯ ВЫБОРКА', 
          fontsize=14, fontweight='bold')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
plt.show()

# Отчет о классификации
print("\nОТЧЕТ О КЛАССИФИКАЦИИ:")
print(classification_report(y_true, y_pred, digits=4))

# ============================================================================
# 8. ВИЗУАЛИЗАЦИЯ ПРИМЕРОВ КЛАССИФИКАЦИИ
# ============================================================================
print("\n" + "="*60)
print("8. ВИЗУАЛИЗАЦИЯ ПРИМЕРОВ КЛАССИФИКАЦИИ")
print("="*60)

# Найдем несколько примеров правильной и неправильной классификации
correct_indices = []
incorrect_indices = []

for i in range(len(y_true)):
    if y_true[i] == y_pred[i]:
        if len(correct_indices) < 5:
            correct_indices.append(i)
    else:
        if len(incorrect_indices) < 5:
            incorrect_indices.append(i)
    if len(correct_indices) >= 5 and len(incorrect_indices) >= 5:
        break

# Визуализация правильных классификаций
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for idx, (i, ax) in enumerate(zip(correct_indices, axes[0])):
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Истино: {y_true[i]}, Предсказано: {y_pred[i]}", fontsize=10)
    ax.axis('off')
    
axes[0, 0].set_ylabel('ПРАВИЛЬНО\nРАСПОЗНАННЫЕ', fontsize=12, fontweight='bold')

# Визуализация неправильных классификаций
for idx, (i, ax) in enumerate(zip(incorrect_indices, axes[1])):
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Истино: {y_true[i]}, Предсказано: {y_pred[i]}", fontsize=10, color='red')
    ax.axis('off')
    
axes[1, 0].set_ylabel('НЕПРАВИЛЬНО\nРАСПОЗНАННЫЕ', fontsize=12, fontweight='bold')

plt.suptitle('ПРИМЕРЫ РАСПОЗНАВАНИЯ ЦИФР', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 9. ВЫВОДЫ И РЕЗУЛЬТАТЫ
# ============================================================================
print("\n" + "="*80)
print("ВЫВОДЫ И РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ ВАРИАНТА 2")
print("="*80)

print("\n1. РАЗБИЕНИЕ ВЫБОРКИ:")
print("   - Исходная обучающая выборка: 60 000 изображений")
print("   - После разбиения:")
print("     • Обучающая выборка: 50 000 изображений")
print("     • Валидационная выборка: 10 000 изображений")
print("     • Тестовая выборка: 10 000 изображений")
print("   - Разбиение выполнено случайным образом с сохранением распределения классов")

print("\n2. АРХИТЕКТУРА НЕЙРОННОЙ СЕТИ:")
print("   - Входной слой: 784 нейрона (28×28 пикселей)")
print("   - Скрытые слои: 256 → 128 → 64 нейронов с ReLU-активацией")
print("   - Dropout-слои для предотвращения переобучения (0.3, 0.3, 0.2)")
print("   - Выходной слой: 10 нейронов с softmax-активацией")

print("\n3. РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
print(f"   - Обучение завершено за {training_time:.2f} секунд")
print(f"   - Количество эпох обучения: {len(history.history['loss'])}")
print(f"   - Финальная точность на обучающей выборке: {history.history['accuracy'][-1]:.4f}")
print(f"   - Финальная точность на валидационной выборке: {history.history['val_accuracy'][-1]:.4f}")

print("\n4. РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
print(f"   - Точность на тестовой выборке: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   - Потери на тестовой выборке: {test_loss:.4f}")
if test_accuracy >= 0.97:
    print(f"   - ✓ ТРЕБОВАНИЕ ВЫПОЛНЕНО: точность ≥ 97%")
else:
    print(f"   - ⚠ ТРЕБОВАНИЕ НЕ ВЫПОЛНЕНО: точность < 97%")

print("\n5. АНАЛИЗ ПЕРЕОБУЧЕНИЯ:")
val_acc = history.history['val_accuracy'][-1]
train_acc = history.history['accuracy'][-1]
gap = train_acc - val_acc
print(f"   - Разница между точностью на обучающей и валидационной выборках: {gap:.4f}")
if gap > 0.05:
    print("   - ⚠ Признаки переобучения: значительная разница (>0.05)")
else:
    print("   - ✓ Переобучение контролируется эффективно")

print("\n" + "="*80)
print("ЗАДАНИЕ ВАРИАНТА 2 УСПЕШНО ВЫПОЛНЕНО!")
print("="*80)

# Сохранение модели
model.save('mnist_model_variant2.h5')
print("\nМодель сохранена в файл: mnist_model_variant2.h5")