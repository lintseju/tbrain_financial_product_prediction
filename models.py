import logging
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

import constants as c


class Model:

    def __init__(self, model_name, params, feat_cols):
        self.model_name = model_name
        self.model_class = locals().get(model_name, globals().get(model_name))
        self.params = params
        self.best_params = None
        self.models = [None for _ in range(4)]
        self.feat_cols = feat_cols
        self.pos_rate = np.zeros(4)
        self.feat_threshold = 2

    def cross_validation(self, feature, label, k_fold=3):
        best_score = [-1.0, -1.0, -1.0, -1.0]
        best_train_score = [0.0, 0.0, 0.0, 0.0]
        best_params = [dict(), dict(), dict(), dict()]
        for i, l in enumerate(c.LABELS):
            feature_i = feature[i]
            nz_rows = np.where(np.abs(feature_i[:, :len(self.feat_cols[i])]).sum(axis=1) >= self.feat_threshold)[0]
            feature_i = feature_i[nz_rows]
            label_i = label[nz_rows, i]
            for param in self.params:
                train_score = []
                valid_score = []
                for j in range(k_fold):
                    logging.info('Fold %d', j)
                    valid_index = np.arange(len(label_i))[j::3]
                    train_index = sorted(set(np.arange(len(label_i))) - set(valid_index))
                    train_feature = feature_i[train_index]
                    train_label = label_i[train_index]
                    valid_feature = feature_i[valid_index]
                    valid_label = label_i[valid_index]

                    if self.model_name == 'XGBClassifier':
                        param['scale_pos_weight'] = (len(train_label) - sum(train_label)) / sum(train_label)
                    model = self._init_model(param)
                    self._train_one(model, train_feature, train_label)
                    train_pred_j = self._predict_one(model, train_feature)
                    train_score.append(f1_score(train_label, train_pred_j))

                    valid_pred_j = self._predict_one(model, valid_feature)
                    valid_score.append(f1_score(valid_label, valid_pred_j))
    
                train_score = np.mean(train_score)
                valid_score = np.mean(valid_score)
                if valid_score > best_score[i]:
                    best_score[i] = valid_score
                    best_train_score[i] = train_score
                    best_params[i] = param
                logging.info('param: %s', str(param))
                logging.info('Train F1 %f, Valid F1 %f', train_score, valid_score)
    
            logging.info('Label %s Best param: %s', l, str(best_params[i]))
            logging.info('Label %s Best Train F1 %f, Valid F1 %f', l, best_train_score[i], best_score[i])
        logging.info('Train score %f, Valid score %f',
                     sum(best_train_score * c.LABEL_WEIGHT),
                     sum(best_score * c.LABEL_WEIGHT))
        self.best_params = best_params

    def _init_model(self, param):
        return self.model_class(**param)

    def _train_one(self, model, feature, label):
        model.fit(feature, label)

    def train(self, feature, label):
        for i in range(len(c.LABELS)):
            feature_i = feature[i]
            nz_rows = np.where(np.abs(feature_i[:, :len(self.feat_cols[i])]).sum(axis=1) >= self.feat_threshold)[0]
            feature_i = feature_i[nz_rows]
            label_i = label[nz_rows, i]

            logging.info('train shape %s, train positive %d (%.2f%%)', feature_i.shape, label_i.sum(),
                         label_i.sum() / len(label_i) * 100)
            if self.model_name == 'XGBClassifier':
                self.best_params[i]['scale_pos_weight'] = (len(label_i) - sum(label_i)) / sum(label_i)
            self.models[i] = self._init_model(self.best_params[i])
            self._train_one(self.models[i], feature_i, label_i)
            self.pos_rate[i] = sum(label_i) / len(label_i)

    def predict(self, feature):
        pred = []
        for i in range(len(c.LABELS)):
            logging.info('test shape %s', feature[i].shape)
            pred.append(self._predict_one(self.models[i], feature[i]))
        return np.vstack(pred).T

    def predict_proba(self, feature):
        pred = []
        for i in range(len(c.LABELS)):
            logging.info('test shape %s', feature[i].shape)
            pred.append(self._predict_one(self.models[i], feature[i], predict_proba=True))
        return np.vstack(pred).T

    def _predict_one(self, model, feature, predict_proba=False):
        pred = model.predict_proba(feature)[:, 1]
        if not predict_proba:
            pred = np.round(pred)
        return pred
