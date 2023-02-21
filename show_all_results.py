import os, sys
import argparse
from os.path import basename, dirname
import pandas as pd
from constants import *

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--dataset',
        type = str, required = True,
        help='Dataset (csv file)')
    parser.add_argument(
        '-q', '--qualify',
        choices = quality_metrics,
        help = 'Metric For Rules Qualification. Choices: ' + str(quality_metrics),
        type = str.lower)
    parser.add_argument(
        '-c', '--column',
        help = 'Column For Metrics Classification',
        type = str.lower, default = 'mcc')
    parser.add_argument(
        '-s', '--support',
        help = 'Support',
        type = float)
    args = parser.parse_args(argv)
    return args

def result_directories(args, supp, qfy, th):
    # - ->> directories to save data <<- -#
    root_path = os.path.dirname(os.path.realpath(__file__))
    dir_name = basename(args.dataset)
    dir_name = dir_name.split('.')[0]
    dir_name += f'_S{int(supp * 100.0)}'
    dir_name += '_C95'
    dir_name += '_L4'

    path = os.path.join(root_path, 'run_files', dir_name)
    DIR_BASE = path

    path = os.path.join(path, qfy)
    DIR_QFY = path

    th_dir = f'T{int(th * 100.0)}'
    DIR_TH = os.path.join(path, th_dir)

    return DIR_BASE, DIR_QFY, DIR_TH

if __name__=="__main__":
    args = parse_args(sys.argv[1:])

    results = pd.DataFrame()
    ss = [args.support] if args.support else [0.1, 0.2, 0.3]
    for s in ss:
        qs = [args.qualify] if args.qualify else quality_metrics
        for q in qs:
            ts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for t in ts:
                DIR_BASE, DIR_QFY, DIR_TH = result_directories(args, s, q, t)
                if os.path.exists(DIR_TH):
                    result = pd.read_csv(os.path.join(DIR_TH, 'general_eval_metrics.csv'))
                    result['supp'] = s
                    result['qfy'] = q
                    result['th'] = t
                    results = pd.concat([results, result], ignore_index = True)
    results.sort_values(by=[args.column], ascending=False, inplace = True)
    print(results.head(15))
