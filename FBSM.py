import numpy as np

class FBSM():

    def __init__(self, user_like, user_dislike, item_features, regularization = 0.1126, learning_rate = 0.001, rank_correlation = 3):
        self.user_like = user_like
        self.user_dislike = user_dislike
        self.item_features = item_features
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.rank_correlation = rank_correlation
        self.D = np.diag(np.random.rand(np.shape(self.item_features)[1]))
        self.V = np.random.rand(np.shape(self.item_features)[1], self.rank_correlation)

    def cal_gradient_D(self, like_item_index, dislike_item_index, user_feature):
        feature_difference = self.item_features[like_item_index] - self.item_features[dislike_item_index]
        result = np.zeros(len(user_feature))
        for i in range(len(user_feature)):
            result[i] = feature_difference[i] * user_feature[i] - self.item_features[like_item_index][i] * self.item_features[like_item_index][i]
        return result

    def cal_gradient_V(self, like_item_index, dislike_item_index, user_feature):
        feature_difference = self.item_features[like_item_index] - self.item_features[dislike_item_index]
        result = np.zeros((len(user_feature), self.rank_correlation))
        for i in range(len(user_feature)):
            result[i] = feature_difference[i] * np.dot(user_feature, self.V) + user_feature[i] * np.dot(feature_difference, self.V) - 2 * \
                        self.item_features[like_item_index][i] * np.dot(self.item_features[like_item_index], self.V)
        return result

    def train(self, nIters = 100):
        for Iter in range(nIters):
            for user_index in range(len(self.user_like)):
                user_feature = 0
                for item in self.user_like[user_index]:
                    user_feature += self.item_features[item]
                np.random.shuffle(self.user_like[user_index])
                np.random.shuffle(self.user_dislike[user_index])
                for like_item_index in self.user_like[user_index]:
                    for dislike_item_index in self.user_dislike[user_index]:
                        relative_rank = self.predict(user_index, like_item_index) - self.predict(user_index, dislike_item_index)
                        gradient_D = self.cal_gradient_D(like_item_index, dislike_item_index, user_feature)
                        gradient_V = self.cal_gradient_V(like_item_index, dislike_item_index, user_feature)
                        tau = np.exp(-relative_rank)/(1 + np.exp(-relative_rank))
                        self.D = self.D + self.learning_rate*(tau*gradient_D - 2*self.regularization*self.D)
                        for index in range(len(user_feature)):
                            self.V[index] = self.V[index] + self.learning_rate*(tau*gradient_V[index] - 2*self.regularization*self.V[index])

    def sim(self, i, j):
        return np.dot(np.dot(self.item_features[i], self.D + np.dot(self.V, self.V.T)),self.item_features[j])

    def predict(self, user_index, item_index):
        result = 0
        for item in self.user_like[user_index]:
            if item == item_index:
                continue
            else:
                result += self.sim(item, item_index)
        return result

    def show_D(self):
        return self.D

    def show_V(self):
        return self.V






