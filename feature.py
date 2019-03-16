import argparse
import logging
import numpy as np
import os
import pandas as pd

import constants as c


def rm_host(page):
    start_index = 27 if page.startswith('http://www.esunbank.com.tw/') else 28
    page = page[start_index:]
    return page[:-1] if page[-1] == '/' else page


def get_page_token_list(df_ret):
    page_cnt = dict()

    for row in df_ret.itertuples():
        page = row.page
        if page not in page_cnt:
            page_cnt[page] = 1
        else:
            page_cnt[page] += 1

    return sorted([x[0] for x in page_cnt.items() if x[1] > 10])


def time_check(date, offset, interval):
    return c.BASE_DATE + offset <= date < c.BASE_DATE + offset + interval


def is_label_ind(df, offset):
    return df['TXN_DT'].map(lambda x: time_check(x, offset, 30))


def is_feat_ind(df, offset, col='VISITDATE', interval=30):
    return df[col].map(lambda x: time_check(x, offset, interval))


def mk_label(df_fx, df_cc, df_wm, df_ln):
    df = pd.merge(df_fx[['FX_IND']], df_cc[['CC_IND']], left_index=True, right_index=True, how='outer')
    df = pd.merge(df, df_wm[['WM_IND']], left_index=True, right_index=True, how='outer')
    df = pd.merge(df, df_ln[['LN_IND']], left_index=True, right_index=True, how='outer')
    df = df.fillna(0)
    df = df[~df.index.duplicated(keep='first')]
    return df


def mk_action_feat(df_fx, df_cc, df_wm, df_ln, days):
    df = pd.merge(df_fx[['FX_IND']], df_cc[['CC_IND']], left_index=True, right_index=True, how='outer')
    df = pd.merge(df, df_wm[['WM_IND']], left_index=True, right_index=True, how='outer')
    df = pd.merge(df, df_ln[['LN_IND']], left_index=True, right_index=True, how='outer')
    df = df.fillna(0)
    df = df.groupby(df.index).sum()
    for col in df.columns:
        df[col] = np.sign(df[col].values)
    new_cols = ['ACTION_%d_' % days + col.replace('IND', 'CNT') for col in df.columns]
    df.columns = new_cols
    return df


def mk_ret_feat(df_ret, feature_dict, days):
    id_list = sorted(set(df_ret['CUST_NO']))
    id_dict = dict((x, i) for i, x in enumerate(id_list))

    page_list = feature_dict['page']
    page_dict = dict((x, i) for i, x in enumerate(page_list))

    data = np.zeros((len(id_dict), len(page_list)))
    for row in df_ret.itertuples():
        page = row.page
        if page in page_dict:
            data[id_dict[row.CUST_NO], page_dict[page]] += 1.0

    cols = ['RET_' + str(days) + '_' + x for x in page_list]
    nz_cols = np.where(np.abs(data).sum(axis=0) > 0)[0]
    df = pd.DataFrame(data[:, nz_cols], index=id_list, columns=np.array(cols)[nz_cols])
    df.index.name = 'CUST_NO'
    return df


def append_ret_feat(df_ret, feature_dict):
    page_list = feature_dict['page']
    for page in page_list:
        df_ret['RET_' + page] = 0
        for day, weight in ((3, 16), (7, 8), (15, 4), (30, 1)):
            col = 'RET_' + str(day) + '_' + page
            if col in df_ret.columns:
                df_ret['RET_' + page] += df_ret[col] * weight
    return df_ret


def get_act_dict(df_act, df_action_list):
    act_dict = [[set() for _ in range(4)] for _ in range(7)]
    for row in df_act.itertuples():
        for i, l in enumerate(c.LABELS):
            val = getattr(row, l.replace('_IND', '_RECENT_DT'))
            if np.isnan(val):
                continue
            for j in range(7):
                if c.BASE_DATE + j * 15 + c.FIRST_LABEL_DATE - 60 <= val < c.BASE_DATE + j * 15 + c.FIRST_LABEL_DATE:
                    act_dict[j][i].add(row.Index)

    for i, df_action in enumerate(df_action_list):
        for row in df_action.itertuples():
            for j in range(7):
                if c.BASE_DATE + j * 15 + c.FIRST_LABEL_DATE - 60 <= row.TXN_DT < c.BASE_DATE + j * 15 + c.FIRST_LABEL_DATE:
                    act_dict[j][i].add(row.Index)

    return act_dict


def add_act_feat(df, act):
    cust_nos = set(df.index)
    for i, col in enumerate(['ACTION_60_FX_CNT', 'ACTION_60_CC_CNT', 'ACTION_60_WM_CNT', 'ACTION_60_LN_CNT']):
        df.loc[cust_nos & act[i], col] = 1
    return df.fillna(0)


def mk_performance(df_fx, df_cc, df_wm, df_ln, df_ret):

    def mk_table(table, df, idx):
        for row in df.itertuples():
            if row.TXN_DT >= c.BASE_DATE + c.FIRST_LABEL_DATE + 60:
                continue
            if row.Index not in table:
                table[row.Index] = [set(), set(), set(), set()]
            table[row.Index][idx].add(row.TXN_DT)
        return table

    table = dict()
    for i, df_i in enumerate([df_fx, df_cc, df_wm, df_ln]):
        table = mk_table(table, df_i, i)

    page_dict = dict()
    for row in df_ret.itertuples():
        page = row.page
        if page not in page_dict:
            page_dict[page] = np.zeros(5)
        if row.CUST_NO in table:
            table_i = table[row.CUST_NO]
            for i in range(4):
                for day in table_i[i]:
                    if row.VISITDATE < day <= row.VISITDATE + 30:
                        page_dict[page][i] += 1
                        break
        page_dict[page][4] += 1

    for page in page_dict:
        page_dict[page][:4] /= page_dict[page][4]

    return page_dict


def feat2yml(file, feat_cols):
    with open(file, 'w') as f:
        f.write('feat_cols:\n')
        for col in sorted(feat_cols):
            f.write('  - %s\n' % col)


def get_page_feat(page_dict, idx, n_page, reverse=True):
    feats2rate = dict()
    for page, values in page_dict.items():
        # skip page too small
        if values[4] <= 10:
            continue
        feats2rate[page] = values[idx]
    return [x[0] for x in sorted(feats2rate.items(), key=lambda x: x[1], reverse=reverse)[:n_page]]


def mk_feat_cols(page_dict, n_page):
    ret_days = [
        (3, 5, 10, 15, 30),
        (3, 5, 10, 15, 30),
        (3, 5, 10, 15),
        (3, 5)
    ]
    for i, l in enumerate(c.LABELS):
        feat_cols = []
        for col in c.ACTION_FEAT_COLS[i]:
            for day in c.ACTION_DAYS[i]:
                feat_cols.append('ACTION_%d_%s_CNT' % (day, col))

        used_page = get_page_feat(page_dict, i, n_page)
        for page in used_page:
            for day in ret_days[i]:
                col = 'RET_%d_%s' % (day, page)
                feat_cols.append(col)

        feat2yml('conf/f_%s.yml' % l.lower(), feat_cols)


def main():
    args = get_args()
    logging.info('%s', args)
    logging.info('Start')
    df_ret = pd.read_csv(c.RET_CSV)
    df_ret['page'] = df_ret['PAGE'].map(lambda x: rm_host(x))

    df_act = pd.read_csv(c.ACT_CSV, index_col='CUST_NO')

    df_fx = pd.read_csv(c.FX_CSV, index_col='CUST_NO')
    df_cc = pd.read_csv(c.CC_CSV, index_col='CUST_NO')
    df_wm = pd.read_csv(c.WM_CSV, index_col='CUST_NO')
    df_ln = pd.read_csv(c.LN_CSV, index_col='CUST_NO')
    df_fx['FX_IND'] = 1
    df_cc['CC_IND'] = 1
    df_wm['WM_IND'] = 1
    df_ln['LN_IND'] = 1

    act_dict = get_act_dict(df_act, [df_fx, df_cc, df_wm, df_ln])

    logging.info('Make label')
    df_label_list = []
    for i in range(6):
        df_label_list.append(
            mk_label(
                df_fx[is_label_ind(df_fx, i * 15 + c.FIRST_LABEL_DATE)],
                df_cc[is_label_ind(df_cc, i * 15 + c.FIRST_LABEL_DATE)],
                df_wm[is_label_ind(df_wm, i * 15 + c.FIRST_LABEL_DATE)],
                df_ln[is_label_ind(df_ln, i * 15 + c.FIRST_LABEL_DATE)],
            )
        )
    df_label_list.append(pd.read_csv(c.SAMPLE_CSV, index_col='CUST_NO'))

    logging.info('Make feature')

    page_list = get_page_token_list(df_ret)
    feature_dict = {
        'page': page_list
    }

    df_action_list = []
    for i in range(7):
        df_action_tmp = []
        for day in (30, 15, 7):
            df_action_tmp.append(mk_action_feat(
                df_fx[is_feat_ind(df_fx, i * 15 + 1 + 30 - day, col='TXN_DT', interval=day)],
                df_cc[is_feat_ind(df_cc, i * 15 + 1 + 30 - day, col='TXN_DT', interval=day)],
                df_wm[is_feat_ind(df_wm, i * 15 + 1 + 30 - day, col='TXN_DT', interval=day)],
                df_ln[is_feat_ind(df_ln, i * 15 + 1 + 30 - day, col='TXN_DT', interval=day)],
                day
            ))
        df_action = pd.merge(df_action_tmp[0], df_action_tmp[1], left_index=True, right_index=True, how='left')
        df_action = pd.merge(df_action, df_action_tmp[2], left_index=True, right_index=True, how='left')
        df_action_list.append(df_action)
    del df_action_tmp

    df_ret_feat_list = []
    for i in range(7):
        df_ret_feat_tmp = []
        for day in (30, 15, 10, 5, 3):
            df_ret_feat_tmp.append(
                mk_ret_feat(df_ret[is_feat_ind(df_ret, i * 15 + 1 + 30 - day, interval=day)], feature_dict, day)
            )
        df_ret_feat = pd.merge(df_ret_feat_tmp[0], df_ret_feat_tmp[1], left_index=True, right_index=True, how='left')
        df_ret_feat = pd.merge(df_ret_feat, df_ret_feat_tmp[2], left_index=True, right_index=True, how='left')
        df_ret_feat = pd.merge(df_ret_feat, df_ret_feat_tmp[3], left_index=True, right_index=True, how='left')
        df_ret_feat = pd.merge(df_ret_feat, df_ret_feat_tmp[4], left_index=True, right_index=True, how='left')
        df_ret_feat_list.append(df_ret_feat)
    del df_ret_feat_tmp

    df_feat_list = []
    for i in range(7):
        df_feat = pd.merge(df_action_list[i], df_ret_feat_list[i], left_index=True, right_index=True, how='right')
        df_feat = add_act_feat(df_feat, act_dict[i])
        df_feat_list.append(df_feat)
    del df_action_list
    del df_ret_feat_list

    logging.info('Merge feature and label')
    df_train_list = []
    for i in range(6):
        df_train_list.append(
            pd.merge(df_feat_list[i], df_label_list[i], left_index=True, right_index=True, how='left').fillna(0)
        )
    df_test = pd.merge(df_feat_list[6], df_label_list[6], left_index=True, right_index=True, how='right').fillna(0)
    del df_feat_list
    del df_label_list

    logging.info('Save feature columns')
    f = open('tmp/feature.txt', 'w')
    for col in sorted(df_train_list[0].columns):
        if col in c.LABELS:
            continue
        f.write('  - %s\n' % col)
    f.close()

    if not args.no_yml:
        logging.info('Save feature config')
        page_dict = mk_performance(df_fx, df_cc, df_wm, df_ln, df_ret)
        n_page = int(len(page_dict) * 0.1)
        mk_feat_cols(page_dict, n_page)

    if not args.no_csv:
        logging.info('Save feature csv')
        for i in range(5):
            for col in df_train_list[i].columns:
                df_train_list[i][col] = df_train_list[i][col].replace({0: np.nan})
        for col in df_test.columns:
            df_test[col] = df_test[col].replace({0: np.nan})

        for i in range(5):
            df_train_list[i].to_csv(c.TRAIN_CSV % i)
        df_test.to_csv(c.TEST_CSV)

    logging.info('Done')


def get_args():
    parser = argparse.ArgumentParser(description='Main script')
    parser.add_argument('--no_csv', action='store_true', help='do not save final csv')
    parser.add_argument('--no_yml', action='store_true', help='do not save feature yml')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    os.makedirs(c.TMP_DIR, exist_ok=True)
    main()
