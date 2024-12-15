import pandas as pd
import numpy as np

class NaiveBayes:
    def fit(self, x, y):
        """Train model from train dataset"""
        self.x_ = pd.DataFrame(x)
        self.y_ = pd.DataFrame(y)
        self.n_ = len(x)

        self.features_ = self.x_.columns
        self.labels_ = self.y_.columns
        self.classes_: np.ndarray = np.array([self.y_[lab].unique() for lab in self.labels_], dtype=object)

        self.prior_y = []
        self.x_mean = []
        self.x_std = []

        data_train = pd.concat([self.x_.reset_index(drop=True), self.y_.reset_index(drop=True)], axis=1)
        for lab in self.labels_:
            prior_temp = []
            mean_temp = []
            std_temp = []

            data_train_groupped = data_train.groupby(self.y_[lab].values.squeeze())
            for i, data in data_train_groupped:
                prob_prior = len(data)/self.n_
                mean_x = self.__mean(data)
                std_x = self.__standarddev(data)

                prior_temp.append(prob_prior)
                mean_temp.append(mean_x)
                std_temp.append(std_x)

            self.prior_y.append(prior_temp)
            self.x_mean.append(mean_temp)
            self.x_std.append(std_temp)

        self.prior_y = np.array(self.prior_y)
        self.x_mean = np.array(self.x_mean)
        self.x_std = np.array(self.x_std)

        return self

    def predict(self, x):
        """Predict for a dataset"""
        x = pd.DataFrame(x)
        self.prob_all = []

        # Calculate the probability of every instance
        for i, lab in enumerate(self.labels_):
            prob_lab = []
            for j, uniq in enumerate(self.classes_[i]):
                gauss = 1
                for k, col in enumerate(self.features_):
                    gauss = gauss * self.__gaussian(x[col], self.x_mean[i, j, k], self.x_std[i, j, k])
                prob_lab.append(gauss * self.prior_y[i, j])
            self.prob_all.append(prob_lab)

        self.prob_all = np.array(self.prob_all)

        return self.__classification(self.prob_all)

    def __mean(self, x):
        """Calculate the mean of every column"""
        return self.x_.iloc[x.index].mean(axis=0)

    def __standarddev(self, x):
        """Calculate the standard deviation of every column"""
        return self.x_.iloc[x.index].std(axis=0)

    def __gaussian(self, x, miu, sigma):
        """Calculate the gaussian probability of every instance"""
        return 1/(np.sqrt(2*np.pi*(sigma**2))) * np.exp(-1/2 * ((x-miu)/sigma)**2)

    def __classification(self, proba):
        """Take the result by maximum probability of every instance"""
        final_result = []
        for i, lab in enumerate(self.labels_):
            idx_prob_max = np.argmax(proba[i, :, :], axis=0)
            final = self.classes_[i, :][idx_prob_max]
            final_result.append(final)

        return np.array(final_result).squeeze()
