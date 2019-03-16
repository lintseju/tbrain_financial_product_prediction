import arrow
import logging
import numpy as np
import pandas as pd
import pickle

from sklearn import preprocessing
from sklearn.metrics import f1_score, precision_score, recall_score

import constants as c
import models as m


class Runner:

    def __init__(self, online, model_conf, feat_conf, output_prefix):
        self.train_feature_list = []
        self.train_label_list = []
        self.train_index_list = []
        self.train_label = []
        self.train_pred = []
        self.train_index = []
        self.valid_label = []
        self.valid_pred = []
        self.valid_index = []
        self.test_feature = []
        self.test_pred = []
        self.test_proba_pred = []
        self.test_index = []
        self.online = online
        self.feat_cols = [x['feat_cols'] for x in feat_conf]
        self.model = m.Model(model_conf['model'], model_conf['model_params'], self.feat_cols)
        self.feat_preprocess = model_conf.get('feature_preprocess', [None for _ in range(4)])
        if output_prefix is None:
            self.ts = arrow.utcnow().format('MMDDHHmm')
        else:
            self.ts = output_prefix

    def _feat_preprocess(self, feature, i):
        if 'norm' in self.feat_preprocess[i]:
            feature[i] = preprocessing.normalize(feature[i], axis=0)
        if 'sign' in self.feat_preprocess[i]:
            feature[i] = np.sign(feature[i])
        return feature[i]

    def normalize_feature(self, feature):
        for i in range(4):
            if self.feat_preprocess[i] is not None:
                feature[i] = self._feat_preprocess(feature, i)
        return feature

    def _loaf_df(self, path):
        df = pd.read_csv(path, index_col='CUST_NO').fillna(0)
        for cols in self.feat_cols:
            for col in cols:
                if col not in df.columns:
                    df[col] = 0.0
        return df

    def _load(self):
        for i in (0, 2, 4):
            df = self._loaf_df(c.TRAIN_CSV % i)
            self.train_feature_list.append(self.normalize_feature([df[cols].values for cols in self.feat_cols]))
            self.train_label_list.append(df[c.LABELS].values)
            self.train_index_list.append(df.index.tolist())

        if self.online:
            df = self._loaf_df(c.TEST_CSV)
            self.test_feature = self.normalize_feature([df[cols].values for cols in self.feat_cols])
            self.test_index = df.index.tolist()

    @staticmethod
    def print_score(dataset, label, pred):
        f1 = 0.0
        for i, w in enumerate(c.LABEL_WEIGHT):
            f1_i = f1_score(label[:, i], np.round(pred)[:, i])
            pr_i = precision_score(label[:, i], np.round(pred)[:, i])
            re_i = recall_score(label[:, i], np.round(pred)[:, i])
            f1 += w * f1_i
            logging.info('%s %s precision / recall / f1: %f / %f / %f', c.LABELS[i], dataset, pr_i, re_i, f1_i)
        logging.info('%s f1: %f', dataset, f1)

    def run(self, quite):
        logging.info('Load dataframes')
        self._load()

        feature = []
        self.train_label = np.vstack((self.train_label_list[:2]))

        logging.info('Cross validation')
        for i in range(4):
            feature.append(np.vstack([x[i] for x in self.train_feature_list[:2]]))
        valid_feature = self.train_feature_list[2]
        self.model.cross_validation(feature, self.train_label)

        logging.info('Train')
        self.model.train(feature, self.train_label)

        logging.info('Predict')
        self.train_pred = self.model.predict_proba(feature)
        self.valid_pred = self.model.predict_proba(valid_feature)
        self.valid_label = self.train_label_list[2]
        self.valid_index = self.train_index_list[2]

        Runner.print_score('train', self.train_label, self.train_pred)
        Runner.print_score('validation', self.valid_label, self.valid_pred)

        if self.online:
            feature = []
            for i in range(4):
                feature.append(np.vstack([x[i] for x in self.train_feature_list[:3]]))
            test_feature = self.test_feature

            label = np.vstack((self.train_label_list[:3]))
            self.model.train(feature, label)
            self.test_pred = self.model.predict(test_feature)
            self.test_proba_pred = self.model.predict_proba(test_feature)

        if not quite:
            self._save_submission()

    def _save_submission(self):
        pickle.dump(self.model.models, open(c.OUT_DIR + '%s_model.pkl' % self.ts, 'wb'))
        with open(c.OUT_DIR + '%s_train.csv' % self.ts, 'w') as fp:
            fp.write('CUST_NO,FX_IND,CC_IND,WM_IND,LN_IND,FX_PRED,CC_PRED,WM_PRED,LN_PRED\n')
            for (index, pred), ans in zip(zip(self.train_index, self.train_pred), self.train_label):
                pred_str = ','.join([str(x) for x in pred])
                ans_str = ','.join([str(x) for x in ans])
                fp.write('{},{},{}\n'.format(index, ans_str, pred_str))

        with open(c.OUT_DIR + '%s_valid.csv' % self.ts, 'w') as fp:
            fp.write('CUST_NO,FX_IND,CC_IND,WM_IND,LN_IND,FX_PRED,CC_PRED,WM_PRED,LN_PRED\n')
            for (index, pred), ans in zip(zip(self.valid_index, self.valid_pred), self.valid_label):
                pred_str = ','.join([str(x) for x in pred])
                ans_str = ','.join([str(x) for x in ans])
                fp.write('{},{},{}\n'.format(index, ans_str, pred_str))

        if self.online:
            with open(c.OUT_DIR + '%s_test.csv' % self.ts, 'w') as fp:
                fp.write('CUST_NO,FX_IND,CC_IND,WM_IND,LN_IND\n')
                for index, pred in zip(self.test_index, self.test_pred):
                    pred_str = ','.join([str(x) for x in pred])
                    fp.write('{},{}\n'.format(index, pred_str))

            with open(c.OUT_DIR + '%s_test_proba.csv' % self.ts, 'w') as fp:
                fp.write('CUST_NO,FX_IND,CC_IND,WM_IND,LN_IND\n')
                for index, pred in zip(self.test_index, self.test_proba_pred):
                    pred_str = ','.join([str(x) for x in pred])
                    fp.write('{},{}\n'.format(index, pred_str))
