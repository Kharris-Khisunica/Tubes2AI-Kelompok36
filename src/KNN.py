import numpy as np
import pandas as pd

class KNN:
    def __init__(self, k, metric="euclidean", p=None):
        self.k: int = k
        self.metric: str = metric
        if (self.metric == "manhattan"):
            self.p: int = 1
        elif (self.metric == "euclidean"):
            self.p: int = 2
        elif (self.metric == "minkowski"):
            if p == None:
                raise Exception("For minkowski metric, p is required")
            elif p <= 0:
                raise Exception("For minkowski metric, p must be more than 0")
            self.p: int = p

    def fit(self, x, y) -> 'knn':
        """Save data train, label, and classes"""
        self.x_ = pd.DataFrame(x)
        self.y_ = pd.DataFrame(y)

        self.labels_ = self.y_.columns
        self.classes_: np.ndarray = np.array([self.y_[lab].unique() for lab in self.labels_], dtype=object)

        return self

    def predict(self, x) -> np.ndarray:
        """Predict for a dataset"""
        x = pd.DataFrame(x)

        classes_final = []
        distances_all = []

        for instance in x.values:
            if (self.metric == "manhattan"):
                d = self.__manhattan_distance(instance)

            elif (self.metric == "euclidean"):
                d = self.__euclidean_distance(instance)

            elif (self.metric == "minkowski"):
                if self.p == 1:
                    d = self.__manhattan_distance(instance)
                elif self.p == 2:
                    d = self.__euclidean_distance(instance)
                elif self.p > 2:
                    d = self.__minkowski_distance(instance)

            d_df = pd.DataFrame({"distance": d})

            for label in self.labels_:
                d_df["label"] = self.y_[label].values
                d_nearest = d_df.sort_values("distance")     # Sort nearest distance
                clas = d_nearest.iloc[:self.k]               # Take k neighbours
                distances_all.append(d_df["distance"].values)

                class_freq = clas.groupby(["label"]).count().rename(columns={"distance":"count"}) # Class frequency
                check_if_tie = False
                if len(clas) % 2 == 0: # Even
                    if len(class_freq) > 1:
                        check_if_tie = all(i == class_freq.iloc[0].values for i in class_freq.values.squeeze())

                    if check_if_tie: # Tie check
                        weight = clas.groupby(["label"]).sum()            # Sum distance every class
                        final_class = weight.iloc[np.argmin(weight)].name # Take class with nearest distance
                    else:
                        final_class = class_freq.iloc[np.argmax(class_freq)].name

                else: # Odd
                    final_class = class_freq.iloc[np.argmax(class_freq)].name

                classes_final.append(final_class)

        if len(self.labels_) > 1:
            classes_final = np.array(classes_final).reshape((len(x), len(self.labels_))).astype("O")
        else:
            classes_final = np.array(classes_final).astype("O")
        self.distances = np.array(distances_all)

        return classes_final

    def __manhattan_distance(self, x) -> np.ndarray:
        """Calculate distance with Manhattan metric"""
        return np.array(abs(self.x_ - x).sum(axis=1))

    def __euclidean_distance(self, x) -> np.ndarray:
        """Calculate distance with Euclidean metric"""
        return np.array(np.sqrt(((self.x_ - x)**2).sum(axis=1)))

    def __minkowski_distance(self, x) -> np.ndarray:
        """Calculate distance with Minkowski metric"""
        return np.array((abs((knear.x_.values - x)**self.p).sum(axis=1))**(1/self.p))
