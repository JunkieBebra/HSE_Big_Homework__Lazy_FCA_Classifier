import numpy as np
import pandas as pd
import os
from typing import Any, List
from concurrent.futures import ProcessPoolExecutor


class FastClassifierFCAParallel_LESS:
    def __init__(self):
        self.X_train_pos: pd.DataFrame = None
        self.X_train_neg: pd.DataFrame = None
        self.binary_X_train_pos: pd.DataFrame = None
        self.numerical_X_train_pos: np.ndarray = None
        self.binary_X_train_neg: pd.DataFrame = None
        self.numerical_X_train_neg: np.ndarray = None
        self.numerical_cols: pd.Index = None
        self.binary_cols: pd.Index = None
        self.classifier_lengths: List[int] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.X_train_pos = X_train[y_train == 1].copy()
        self.X_train_neg = X_train[y_train == 0].copy()

        self.binary_cols = X_train.select_dtypes(include="bool").columns
        self.numerical_cols = X_train.select_dtypes(exclude="bool").columns

        self.binary_X_train_pos = self.X_train_pos[self.binary_cols]
        self.numerical_X_train_pos = self.X_train_pos[self.numerical_cols].values
        self.binary_X_train_neg = self.X_train_neg[self.binary_cols]
        self.numerical_X_train_neg = self.X_train_neg[self.numerical_cols].values

    def _get_matches_vectorized(
        self,
        train_binary_data: pd.DataFrame,
        train_numerical_data: np.ndarray,
        binary_intersection: pd.Series,
        numerical_lower_bounds: np.ndarray,
        numerical_upper_bounds: np.ndarray,
    ) -> np.ndarray:

        binary_match = (~binary_intersection | train_binary_data).all(axis=1).values

        numerical_match_per_col = (train_numerical_data >= numerical_lower_bounds) & (
            train_numerical_data <= numerical_upper_bounds
        )
        numerical_match = numerical_match_per_col.all(axis=1)

        return binary_match & numerical_match

    def classify_sample(self, sample: pd.Series, max_matches: int = 6) -> int:

        positive_classifiers = 0
        negative_classifiers = 0
        classifier_lengths = []

        sample_binary = sample[self.binary_cols]
        sample_numerical = sample[self.numerical_cols].values

        for _, pos_sample in self.X_train_pos.iterrows():
            binary_intersection = sample_binary & pos_sample[self.binary_cols]
            numerical_train_values = pos_sample[self.numerical_cols].values

            numerical_lower_bounds = np.minimum(
                sample_numerical, numerical_train_values
            )
            numerical_upper_bounds = np.maximum(
                sample_numerical, numerical_train_values
            )

            binary_features_used = binary_intersection.sum()
            numerical_features_used = len(sample_numerical)
            total_features_used = binary_features_used + numerical_features_used
            if total_features_used > max_matches:
                continue

            num_positive = self._get_matches_vectorized(
                self.binary_X_train_pos,
                self.numerical_X_train_pos,
                binary_intersection,
                numerical_lower_bounds,
                numerical_upper_bounds,
            ).sum()
            num_negative = self._get_matches_vectorized(
                self.binary_X_train_neg,
                self.numerical_X_train_neg,
                binary_intersection,
                numerical_lower_bounds,
                numerical_upper_bounds,
            ).sum()

            if num_negative == 0 and num_positive > 1:
                binary_features_used = binary_intersection.sum()
                numerical_features_used = len(self.numerical_cols)
                classifier_length = binary_features_used + numerical_features_used
                classifier_lengths.append(classifier_length)
                positive_classifiers += 1

        for _, neg_sample in self.X_train_neg.iterrows():
            binary_intersection = sample_binary & neg_sample[self.binary_cols]
            numerical_train_values = neg_sample[self.numerical_cols].values
            numerical_lower_bounds = np.minimum(
                sample_numerical, numerical_train_values
            )
            numerical_upper_bounds = np.maximum(
                sample_numerical, numerical_train_values
            )

            binary_features_used = binary_intersection.sum()
            numerical_features_used = len(sample_numerical)
            total_features_used = binary_features_used + numerical_features_used
            if total_features_used > max_matches:
                continue

            num_positive = self._get_matches_vectorized(
                self.binary_X_train_pos,
                self.numerical_X_train_pos,
                binary_intersection,
                numerical_lower_bounds,
                numerical_upper_bounds,
            ).sum()
            num_negative = self._get_matches_vectorized(
                self.binary_X_train_neg,
                self.numerical_X_train_neg,
                binary_intersection,
                numerical_lower_bounds,
                numerical_upper_bounds,
            ).sum()

            if num_positive == 0 and num_negative > 1:
                binary_features_used = binary_intersection.sum()
                numerical_features_used = len(self.numerical_cols)
                classifier_length = binary_features_used + numerical_features_used
                classifier_lengths.append(classifier_length)
                negative_classifiers += 1

        # return 1 if positive_classifiers > negative_classifiers else 0
        result = 1 if positive_classifiers > negative_classifiers else 0
        return {
            "prediction": result,
            "sample_name": sample.name if hasattr(sample, "name") else "unknown",
            "positive_classifiers": positive_classifiers,
            "negative_classifiers": negative_classifiers,
            "classifier_lengths": classifier_lengths,
        }

    def predict(self, X_test: pd.DataFrame, n_jobs: int = -1) -> List[Any]:

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        print(f"Начинаю предсказание для {len(X_test)} сэмплов...", flush=True)

        predictions = []
        all_classifier_lengths = []

        # Режим отладки - последовательное выполнение
        if n_jobs == 1:
            print("Использую последовательный режим (отладка)...", flush=True)
            for idx, (_, sample) in enumerate(X_test.iterrows()):
                result_dict = self.classify_sample(sample)
                prediction = result_dict["prediction"]
                predictions.append(prediction)

                all_classifier_lengths.extend(result_dict["classifier_lengths"])

                print(
                    f"Сэмпл {idx}: предсказание={prediction}, "
                    f"позитивных классификаторов={result_dict['positive_classifiers']}, "
                    f"негативных классификаторов={result_dict['negative_classifiers']}",
                    flush=True,
                )

            if all_classifier_lengths:
                avg_classifier_length = sum(all_classifier_lengths) / len(
                    all_classifier_lengths
                )
            else:
                avg_classifier_length = 0.0

            print(f"\nСредняя длина классификаторов: {avg_classifier_length:.2f}")
            return predictions

        # Многопроцессный режим
        print(f"Использую многопроцессный режим с {n_jobs} workers...", flush=True)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Подготавливаем задачи
            futures = []
            for idx, (_, sample) in enumerate(X_test.iterrows()):
                future = executor.submit(self.classify_sample, sample)
                futures.append((idx, future))

            # Собираем результаты
            for idx, future in futures:
                result_dict = future.result()
                prediction = result_dict["prediction"]
                predictions.append(prediction)

                all_classifier_lengths.extend(result_dict["classifier_lengths"])

                # Выводим информацию в основном процессе
                print(
                    f"Сэмпл {idx}: предсказание={prediction}, "
                    f"позитивных классификаторов={result_dict['positive_classifiers']}, "
                    f"негативных классификаторов={result_dict['negative_classifiers']}",
                    flush=True,
                )

            if all_classifier_lengths:
                avg_classifier_length = sum(all_classifier_lengths) / len(
                    all_classifier_lengths
                )
            else:
                avg_classifier_length = 0.0

            print(f"\nСредняя длина классификаторов: {avg_classifier_length:.2f}")

            return predictions
