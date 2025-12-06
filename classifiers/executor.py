import numpy as np
import pandas as pd
import os 
from typing import Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager 
import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List
from collections import Counter

class FastClassifierFCAParallel:
     
    def __init__(self):
        self.X_train_pos: pd.DataFrame = None
        self.X_train_neg: pd.DataFrame = None
        self.binary_X_train_pos: pd.DataFrame = None
        self.numerical_X_train_pos: np.ndarray = None
        self.binary_X_train_neg: pd.DataFrame = None
        self.numerical_X_train_neg: np.ndarray = None
        self.numerical_cols: pd.Index = None
        self.binary_cols: pd.Index = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.X_train_pos = X_train[y_train == 1].copy()
        self.X_train_neg = X_train[y_train == 0].copy()

        self.binary_cols = X_train.select_dtypes(include='bool').columns
        self.numerical_cols = X_train.select_dtypes(exclude='bool').columns

        self.binary_X_train_pos = self.X_train_pos[self.binary_cols]
        self.numerical_X_train_pos = self.X_train_pos[self.numerical_cols].values
        self.binary_X_train_neg = self.X_train_neg[self.binary_cols]
        self.numerical_X_train_neg = self.X_train_neg[self.numerical_cols].values

    def _get_matches_vectorized(self,
                                train_binary_data: pd.DataFrame,
                                train_numerical_data: np.ndarray,
                                binary_intersection: pd.Series,
                                numerical_lower_bounds: np.ndarray,
                                numerical_upper_bounds: np.ndarray) -> np.ndarray:
        
        binary_match = (~binary_intersection | train_binary_data).all(axis=1).values
        
        numerical_match_per_col = (train_numerical_data >= numerical_lower_bounds) & \
                                  (train_numerical_data <= numerical_upper_bounds)
        numerical_match = numerical_match_per_col.all(axis=1)

        return binary_match & numerical_match

    def classify_sample(self, sample: pd.Series) -> int:
        
        positive_classifiers = 0
        negative_classifiers = 0
        sample_binary = sample[self.binary_cols]
        sample_numerical = sample[self.numerical_cols].values

        for _, pos_sample in self.X_train_pos.iterrows():
            binary_intersection = sample_binary & pos_sample[self.binary_cols]
            numerical_train_values = pos_sample[self.numerical_cols].values
            numerical_lower_bounds = np.minimum(sample_numerical, numerical_train_values)
            numerical_upper_bounds = np.maximum(sample_numerical, numerical_train_values)

            num_positive = self._get_matches_vectorized(
                self.binary_X_train_pos, self.numerical_X_train_pos, 
                binary_intersection, numerical_lower_bounds, numerical_upper_bounds
            ).sum()
            num_negative = self._get_matches_vectorized(
                self.binary_X_train_neg, self.numerical_X_train_neg, 
                binary_intersection, numerical_lower_bounds, numerical_upper_bounds
            ).sum()

            if num_negative == 0 and num_positive > 1:
                positive_classifiers += 1

        for _, neg_sample in self.X_train_neg.iterrows():
            binary_intersection = sample_binary & neg_sample[self.binary_cols]
            numerical_train_values = neg_sample[self.numerical_cols].values
            numerical_lower_bounds = np.minimum(sample_numerical, numerical_train_values)
            numerical_upper_bounds = np.maximum(sample_numerical, numerical_train_values)

            num_positive = self._get_matches_vectorized(
                self.binary_X_train_pos, self.numerical_X_train_pos, 
                binary_intersection, numerical_lower_bounds, numerical_upper_bounds
            ).sum()
            num_negative = self._get_matches_vectorized(
                self.binary_X_train_neg, self.numerical_X_train_neg, 
                binary_intersection, numerical_lower_bounds, numerical_upper_bounds
            ).sum()
            
            if num_positive == 0 and num_negative > 1:
                negative_classifiers += 1

        return 1 if positive_classifiers > negative_classifiers else 0

    def predict(self, X_test: pd.DataFrame, n_jobs: int = -1) -> List[Any]:
        
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        print(n_jobs)
        predictions = []
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(self.classify_sample, sample) 
                       for _, sample in X_test.iterrows()]
            
            for future in futures:
                predictions.append(future.result())
            
        return predictions
    
    def calculate_frequency_importance(self, X_test: pd.DataFrame, n_jobs: int = -1) -> Dict[str, pd.DataFrame]:
        """
        Вычисляет частотную важность признаков, используя внешнюю параллельную функцию.
        """
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        
        # Передаем только необходимые данные, избегая Manager.dict
        pos_feature_counts = Counter()
        neg_feature_counts = Counter()
        total_pos = 0
        total_neg = 0
        
        # 1. Сбор аргументов для передачи в каждый процесс
        common_args = (
            self.X_train_pos, self.X_train_neg,
            self.binary_X_train_pos, self.numerical_X_train_pos,
            self.binary_X_train_neg, self.numerical_X_train_neg,
            self.binary_cols, self.numerical_cols
        )
        
        # 2. Формирование списка задач (образец + общие данные)
        tasks = [(sample_row, *common_args) for _, sample_row in X_test.iterrows()]

        # 3. Запуск параллельного выполнения
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # map вызывает _fca_count_worker_func для каждого элемента в tasks
            futures = [executor.submit(_fca_count_worker_func, *task) for task in tasks]

            # 4. Агрегация результатов
            for future in as_completed(futures):
                try:
                    local_pos_features, local_neg_features, local_pos_rules_count, local_neg_rules_count = future.result()
                    
                    pos_feature_counts.update(local_pos_features)
                    neg_feature_counts.update(local_neg_features)
                    total_pos += local_pos_rules_count
                    total_neg += local_neg_rules_count
                    
                except Exception as e:
                    print(f"Ошибка в дочернем процессе: {e}")
                    raise e
            
        print(f"Общее количество найденных чистых POSITIVE правил: {total_pos}")
        print(f"Общее количество найденных чистых NEGATIVE правил: {total_neg}")

        # --- Расчет долей (нормировка) ---
        
        # ... (Код для расчета долей и создания DataFrame остается прежним) ...

        if total_pos > 0:
            pos_importance = {feat: count / total_pos for feat, count in pos_feature_counts.items()}
        else:
            # Инициализация всех признаков нулями, если нет правил
            pos_importance = {feat: 0 for feat in self.X_train_pos.columns} 
            
        if total_neg > 0:
            neg_importance = {feat: count / total_neg for feat, count in neg_feature_counts.items()}
        else:
            neg_importance = {feat: 0 for feat in self.X_train_neg.columns}

        # Преобразование в DataFrame для удобного сравнения
        df_pos = pd.DataFrame(
            {'Feature Importance (Pos)': pos_importance}
        ).sort_values(by='Feature Importance (Pos)', ascending=False)
        
        df_neg = pd.DataFrame(
            {'Feature Importance (Neg)': neg_importance}
        ).sort_values(by='Feature Importance (Neg)', ascending=False)

        return {
            'positive': df_pos,
            'negative': df_neg,
            'total_pos_rules': total_pos,
            'total_neg_rules': total_neg,
        }

# --- РАБОЧАЯ ФУНКЦИЯ ДЛЯ ПАРАЛЛЕЛИЗАЦИИ (Должна быть вне класса) ---

def _fca_count_worker_func(
        sample_row: pd.Series,
        X_train_pos: pd.DataFrame, X_train_neg: pd.DataFrame,
        binary_X_train_pos: pd.DataFrame, numerical_X_train_pos: np.ndarray,
        binary_X_train_neg: pd.DataFrame, numerical_X_train_neg: np.ndarray,
        binary_cols: pd.Index, numerical_cols: pd.Index
    ):
    """
    Чистая функция, которая безопасно выполняется в дочерних процессах.
    Возвращает локальные счетчики правил и признаков для одного тестового образца.
    """
    from collections import Counter # Импорт внутри для надежности pickling
    
    # Внутренняя (picklable) версия логики _get_matches_vectorized
    def get_matches_vectorized(train_binary_data, train_numerical_data, binary_intersection, numerical_lower_bounds, numerical_upper_bounds):
        binary_match = (~binary_intersection | train_binary_data).all(axis=1).values
        numerical_match_per_col = (train_numerical_data >= numerical_lower_bounds) & \
                                  (train_numerical_data <= numerical_upper_bounds)
        numerical_match = numerical_match_per_col.all(axis=1)
        return binary_match & numerical_match
    
    # Внутренняя (picklable) версия логики _add_features_to_counter
    def add_features_to_counter(counter: Counter, binary_cols: pd.Index, numerical_cols: pd.Index, binary_mask: pd.Series, num_lower: np.ndarray, num_upper: np.ndarray):
        active_binary = binary_cols[binary_mask].tolist()
        counter.update(active_binary)
        for col in numerical_cols:
            counter.update([col]) # Считаем, что все числовые признаки участвуют в правиле

    # Инициализация
    sample_binary = sample_row[binary_cols]
    sample_numerical = sample_row[numerical_cols].values
    
    local_pos_rules_count = 0
    local_neg_rules_count = 0
    local_pos_features = Counter()
    local_neg_features = Counter()

    # --- 1. Поиск Положительных правил (Логика скопирована из _count_features_in_sample) ---
    for _, pos_sample in X_train_pos.iterrows():
        binary_intersection = sample_binary & pos_sample[binary_cols]
        numerical_train_values = pos_sample[numerical_cols].values
        numerical_lower = np.minimum(sample_numerical, numerical_train_values)
        numerical_upper = np.maximum(sample_numerical, numerical_train_values)

        neg_support = get_matches_vectorized(
            binary_X_train_neg, numerical_X_train_neg, 
            binary_intersection, numerical_lower, numerical_upper
        ).sum()

        if neg_support == 0:
            pos_support = get_matches_vectorized(
                binary_X_train_pos, numerical_X_train_pos, 
                binary_intersection, numerical_lower, numerical_upper
            ).sum()
            
            if pos_support > 1:
                local_pos_rules_count += 1
                add_features_to_counter(local_pos_features, binary_cols, numerical_cols, binary_intersection, numerical_lower, numerical_upper)

    # --- 2. Поиск Отрицательных правил ---
    for _, neg_sample in X_train_neg.iterrows():
        binary_intersection = sample_binary & neg_sample[binary_cols]
        numerical_train_values = neg_sample[numerical_cols].values
        numerical_lower = np.minimum(sample_numerical, numerical_train_values)
        numerical_upper = np.maximum(sample_numerical, numerical_train_values)

        pos_support = get_matches_vectorized(
            binary_X_train_pos, numerical_X_train_pos, 
            binary_intersection, numerical_lower, numerical_upper
        ).sum()

        if pos_support == 0:
            neg_support = get_matches_vectorized(
                binary_X_train_neg, numerical_X_train_neg, 
                binary_intersection, numerical_lower, numerical_upper
            ).sum()
            
            if neg_support > 1:
                local_neg_rules_count += 1
                add_features_to_counter(local_neg_features, binary_cols, numerical_cols, binary_intersection, numerical_lower, numerical_upper)

    # Возвращаем локальные счетчики, которые будут объединены в главном процессе
    return local_pos_features, local_neg_features, local_pos_rules_count, local_neg_rules_count