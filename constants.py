import numpy as np

FX_CSV = 'data/TBN_FX_TXN.csv'
CC_CSV = 'data/TBN_CC_APPLY.csv'
WM_CSV = 'data/TBN_WM_TXN.csv'
LN_CSV = 'data/TBN_LN_APPLY.csv'
SAMPLE_CSV = 'data/TBN_Y_ZERO.csv'

RET_CSV = 'data/TBN_CUST_BEHAVIOR.csv'
ACT_CSV = 'data/TBN_RECENT_DT.csv'

TRAIN_CSV = 'tmp/train_%d.csv'
TEST_CSV = 'tmp/test.csv'

TMP_DIR = 'tmp/'
OUT_DIR = 'out/'

BASE_DATE = 9447  # start from 9447 + 1
FIRST_LABEL_DATE = 31

LABELS = ['FX_IND', 'CC_IND', 'WM_IND', 'LN_IND']
LABEL_WEIGHT = np.array([1, 10, 20, 20])

ACTION_FEAT_COLS = [
    ['FX', 'WM'],
    ['CC'],
    ['FX', 'WM'],
    ['LN']
]
ACTION_DAYS = [
    (7, 15, 30, 60),
    (15, 30, 60),
    (60,),
    (30, 60)
]
