from abc import ABC, abstractmethod
import gzip
import multiprocessing
from typing import Any, Iterable
from functools import partial
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import pdb


class Compressor(ABC):
    def __init__(self) -> None: ...

    @abstractmethod
    def compress(self, text: str) -> bytes: ...


class GzipCompressor(Compressor):
    def __init__(self) -> None:
        super().__init__()

    def compress(self, text: str):
        return gzip.compress(text.encode())


def regress(
    x1: str,
    X_train: Iterable[str],
    y_train: np.ndarray,
    k: int,
    compressor: Compressor = GzipCompressor(),
) -> Iterable:
    Cx1 = len(compressor.compress(x1))
    distance_from_x1 = []
    for x2 in X_train:
        Cx2 = len(compressor.compress(x2))
        x1x2 = " ".join([x1, x2])
        x2x1 = " ".join([x2, x1])
        Cx1x2 = len(compressor.compress(x1x2))
        Cx2x1 = len(compressor.compress(x2x1))
        ncd = (0.5 * (Cx1x2 + Cx2x1) - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)

    distance_from_x1 = np.array(distance_from_x1)
    sorted_idx = np.argsort(distance_from_x1)
    top_k_values = y_train[sorted_idx[:k]]
    top_k_dists = distance_from_x1[sorted_idx[:k]]

    # print(top_k_values.shape, top_k_dists.shape)

    n_props = top_k_values.shape[1]

    task_preds = []
    for vals, dists in zip(
        np.array(top_k_values).T, np.tile(np.array(top_k_dists), (1, n_props)).T
    ):
        dists = 1 - dists
        task_preds.append(np.mean(vals * dists) / np.sum(dists))

    return task_preds


class ZipRegressor(object):
    def __init__(self) -> "ZipRegressor":
        pass

    def fit_predict(
        self,
        X_train: Iterable[str],
        y_train: Iterable,
        X: Iterable[str],
        k: int = 25,
        compressor: Compressor = GzipCompressor(),
    ) -> np.ndarray:
        preds = []

        y_train = np.array(y_train)

        if len(y_train.shape) == 1:
            y_train = np.expand_dims(y_train, axis=1)

        cpu_count = multiprocessing.cpu_count()

        with multiprocessing.Pool(cpu_count) as p:
            preds = p.map(
                partial(
                    regress,
                    X_train=X_train,
                    y_train=y_train,
                    k=k,
                    compressor=compressor,
                ),
                X,
            )

        return np.array(preds)


class ZipRegressor_CV(object):
    def __init__(self) -> "ZipRegressor_CV":
        pass

    def fit_predict(
        self,
        X_train: Iterable[str],
        y_train: Iterable,
        X: Iterable[str],
        compressor: Compressor = GzipCompressor(),
    ) -> np.ndarray:
        y_train = np.array(y_train)
        if len(y_train.shape) == 1:
            y_train = np.expand_dims(y_train, axis=1)

        # Determine the best k using 5-fold cross-validation
        kf = KFold(n_splits=5)
        k_values = range(1, 35)  # Range of k values to try
        k_performance = {}

        for k in k_values:
            k_scores = []
            for train_index, test_index in kf.split(X_train):
                X_train_fold, X_val_fold = (
                    np.array(X_train)[train_index],
                    np.array(X_train)[test_index],
                )
                y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

                # Train and predict with the current fold and k value
                preds_fold = self._predict_with_k(
                    X_train_fold, y_train_fold, X_val_fold, k, compressor
                )
                fold_score = mean_squared_error(y_val_fold, preds_fold)
                k_scores.append(fold_score)

            k_performance[k] = np.mean(k_scores)

        # Select the k with the lowest average score (MSE)
        best_k = min(k_performance, key=k_performance.get)

        # Make predictions with the best k value
        return best_k, self._predict_with_k(X_train, y_train, X, best_k, compressor)

    def _predict_with_k(self, X_train, y_train, X, k, compressor):
        cpu_count = multiprocessing.cpu_count()
        with multiprocessing.Pool(cpu_count) as p:
            preds = p.map(
                partial(
                    regress,
                    X_train=X_train,
                    y_train=y_train,
                    k=k,
                    compressor=compressor,
                ),
                X,
            )
        return np.array(preds)

if __name__ == "__main__":

    import bz2, pickle
    from vec2str import ZipFeaturizer

    compress_fileopener = {True: bz2.BZ2File, False: open}

    def loadpkl(filename: str, compress: bool = False):
        """
        Load an object from a pickle file.
        filename : name of the imported file
        compress : whether bz2 compression was used in creating the loaded file.
        """
        input_file = compress_fileopener[compress](filename, "rb")
        obj = pickle.load(input_file)
        input_file.close()
        return obj

    data = loadpkl("data/rep_delaney_rdkit.pkl", compress=True)
    converter = ZipFeaturizer()
    X_train = converter.bin_vectors(data["X_train"])
    X_test = converter.bin_vectors(data["X_test"])
    y_train = data["y_train"]
    y_test = data["y_test"]

    # pdb.set_trace()
    reg = ZipRegressor()
    preds = reg.fit_predict(X_train, y_train, X_test)
    # create a scatter plot of the predicted vs actual values
    import matplotlib.pyplot as plt

    plt.scatter(y_test, preds)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()

    #pdb.set_trace()
