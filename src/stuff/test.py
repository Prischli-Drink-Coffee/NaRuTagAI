video = {
    "Спорт": {
        "Волейбол": [
            "Чемпионат среди молодежи", "Пляжный волейбол"
        ],
        "Футбол": [
            "Футбольный мяч", "Европа"
        ]
    },
    "Автомобили": {
        "Формула 1": [
            "Быстрые машины", "Скорость", "Аварии"
        ]
    },
    "Фильмы": {
        "Драма": [
            "Любовь", "Отношения"
        ]
    }
}


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Создаем простую нейронную сеть
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Слой с 2 входами и 2 выходами
        self.fc2 = nn.Linear(2, 1)  # Слой с 2 входами и 1 выходом

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Инициализация весов сети нулями
def init_weights_zero(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight, 0)
        nn.init.constant_(layer.bias, 0)

# Создаем модель
model = SimpleNN()

# Инициализируем веса нулями
model.apply(init_weights_zero)

# Проверим веса
print("Initial weights of the network:")
print("fc1 weight:\n", model.fc1.weight)
print("fc2 weight:\n", model.fc2.weight)

# Используем простой набор данных для обучения (например, логическая операция AND)
X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = torch.tensor([[0.0], [0.0], [0.0], [1.0]])  # Ответы для операции AND

# Определим оптимизатор и функцию потерь
criterion = nn.MSELoss()  # Для простоты используем MSE
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Обучение модели
epochs = 1000
for epoch in range(epochs):
    # Прямой проход
    output = model(X_train)
    loss = criterion(output, y_train)

    # Обратный проход и обновление весов
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Результаты после обучения
print("\nПосле обучения:")
with torch.no_grad():
    output = model(X_train)
    print("Предсказания модели:", output)