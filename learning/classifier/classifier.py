#from sklearn.externals import joblib
import joblib
from sklearn.model_selection import GridSearchCV
import os
import pickle


class Classifier(object):
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        if self.model_file_path is not None and os.path.exists(self.model_file_path):
            self.load(model_file_path)
        else:
            self.model = None

    # def __pipeline(self):
    #     pass

    @property
    def is_trained(self):
        return self.model is not None

    def save(self, file_path):
        joblib.dump(self.model, file_path)
        # pickle.dump(self.model, file_path)

    def load(self, file_path):
        self.model = joblib.load(file_path)
        # with open(file_path, 'rb') as pickle_file:
        #     self.model = pickle.load(pickle_file)

    def train(self, X_train, y_train):
        optimized_classifier = GridSearchCV(self.pipeline, self.parameters, n_jobs=-1, cv=10)
        # optimized_classifier = self.pipeline
        self.model = optimized_classifier.fit(X_train, y_train)
        # print('cv_results_: ', optimized_classifier.cv_results_)
        print('best_score_: ', optimized_classifier.best_score_)
        print('best_params_: ', optimized_classifier.best_params_)
        print('cv_results_: ', optimized_classifier.cv_results_['mean_test_score'])
        if self.model_file_path is not None:
            self.save(self.model_file_path)
        return self.model.best_score_

    def predict(self, X_test):
        if self.is_trained:
            return self.model.predict(X_test)
        else:
            return None

    def predict_proba(self, X_test):
        if self.is_trained:
            return self.model.predict_proba(X_test)
        else:
            return None


