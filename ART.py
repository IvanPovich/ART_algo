import numpy as np

class ART:
    def __init__(self, vigilance=0.8):
        """
        vigilance: Поріг схожості для формування кластера.
        """
        self.vigilance = vigilance
        self.clusters = []  # Список для збереження центрів кластерів

    def compute_simil(self, input_vector, cluster_center):
        """Обчислює схожість між вектором і центром кластера як відношення загальних компонентів."""
        return np.sum(np.minimum(input_vector, cluster_center)) / np.sum(input_vector)

    def train(self, input_vectors):
        """
        Формує кластери на основі схожості з існуючими центрами.
        input_vectors: Масив векторів для кластеризації.
        """
        for input_vector in input_vectors:
            input_vector = np.array(input_vector)
            matched = False
            
            for i, cluster_center in enumerate(self.clusters):
                similarity = self.compute_simil(input_vector, cluster_center)
                
                if similarity >= self.vigilance:
                    # Оновлення центру кластера
                    self.clusters[i] = np.minimum(input_vector, cluster_center)
                    matched = True
                    break

            # Якщо схожого кластера немає, створюється новий
            if not matched:
                self.clusters.append(input_vector)

    def predict(self, input_vector):
        """
        Визначає кластер для вхідного вектора на основі максимальної схожості.
        Повертає індекс найбільш схожого кластера або -1, якщо схожого кластера немає.
        """
        input_vector = np.array(input_vector)
        for i, cluster_center in enumerate(self.clusters):
            similarity = self.compute_simil(input_vector, cluster_center)
            if similarity >= self.vigilance:
                return i  # Індекс найбільш схожого кластера

        return -1  # Якщо не знайдено відповідного кластера

# Приклад використання
art = ART(vigilance=0.8)

# Тренування на векторах, які належать до двох явно різних кластерів
input_vectors = [
    [0.9, 0.9],   # Яскраво виражений перший кластер
    [0.85, 0.85],
    [0.1, 0.1],   # Яскраво виражений другий кластер
    [0.15, 0.2],
    [0.88, 0.9],  # Ближче до першого кластера
    [0.2, 0.15]   # Ближче до другого кластера
]
art.train(input_vectors)

# Класифікація нового вектора
new_vector_1 = [0.86, 0.88]
new_vector_2 = [0.12, 0.18]
cluster_index_1 = art.predict(new_vector_1)
cluster_index_2 = art.predict(new_vector_2)

print(f"Вектор {new_vector_1} відноситься до кластера {cluster_index_1}")
print(f"Вектор {new_vector_2} відноситься до кластера {cluster_index_2}")

# Вивід центрів кластерів для візуалізації
# Вивід центрів кластерів у форматі списків з округленням
# Спершу округлення, потім перетворення на списки
rounded_clusters = [np.round(cluster, 2).tolist() for cluster in art.clusters]
print("Центри кластерів:", rounded_clusters)


