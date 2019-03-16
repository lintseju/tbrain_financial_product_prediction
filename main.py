import argparse
import logging
import os
import yaml

import constants as c
import runner as r


def get_args():
    parser = argparse.ArgumentParser(description='Main script')
    parser.add_argument('--online', action='store_true', help='run online version')
    parser.add_argument('-q', '--quite', action='store_true', help="don't save to disk")
    parser.add_argument('-m', '--model', type=str, default='conf/m_lgbm.yml', help='model parameters')
    parser.add_argument('-o', '--output', type=str, default=None, help='output prefix')
    parser.add_argument('-f', '--feature_fx', type=str, default='conf/f_fx_ind.yml', help='feature columns')
    parser.add_argument('-c', '--feature_cc', type=str, default='conf/f_cc_ind.yml', help='feature columns')
    parser.add_argument('-w', '--feature_wm', type=str, default='conf/f_wm_ind.yml', help='feature columns')
    parser.add_argument('-l', '--feature_ln', type=str, default='conf/f_ln_ind.yml', help='feature columns')
    return parser.parse_args()


def main():
    args = get_args()
    logging.info('%s', args)
    model_conf = yaml.load(open(args.model, 'r'))
    feat_fx_conf = yaml.load(open(args.feature_fx, 'r'))
    feat_cc_conf = yaml.load(open(args.feature_cc, 'r'))
    feat_wm_conf = yaml.load(open(args.feature_wm, 'r'))
    feat_ln_conf = yaml.load(open(args.feature_ln, 'r'))

    runner = r.Runner(args.online, model_conf, [feat_fx_conf, feat_cc_conf, feat_wm_conf, feat_ln_conf], args.output)
    runner.run(args.quite)

    logging.info('Done')


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    os.makedirs(c.OUT_DIR, exist_ok=True)
    main()
