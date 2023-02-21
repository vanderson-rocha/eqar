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
        type = str.lower, required = True)
    parser.add_argument(
        '-s', '--min-support',
        help = 'Minimum Support (must be > 0.0 and < 1.0). Default: 0.1',
        type = float, default = 0.1)
    parser.add_argument(
        '--allth', help = 'Execute All Threshold',
        action = 'store_true')
    args = parser.parse_args(argv)
    return args

def result_directories(args):
    # - ->> directories to save data <<- -#
    root_path = os.path.dirname(os.path.realpath(__file__))
    dir_name = basename(args.dataset)
    dir_name = dir_name.split('.')[0]
    dir_name += f'_S{int(args.min_support * 100.0)}'
    dir_name += '_C95'
    dir_name += '_L4'

    path = os.path.join(root_path, 'run_files', dir_name)
    DIR_BASE = path

    path = os.path.join(path, args.qualify)
    DIR_QFY = path

    return DIR_BASE, DIR_QFY

if __name__=="__main__":
    args = parse_args(sys.argv[1:])

    DIR_BASE, DIR_QFY = result_directories(args)


    biggest_mcc = 0.0
    count = 0
    mt = 0.1
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #0.1, 0.2, 0.3, 0.4,
        cmd = f'python3 run.py -d {args.dataset} -s {args.min_support} -q {args.qualify} -t {t}'# --verbose'
        print(f'Running {cmd}')
        os.system(cmd)
        th_dir = f'T{int(t * 100.0)}'
        DIR_TH = os.path.join(DIR_QFY, th_dir)
        result = pd.read_csv(os.path.join(DIR_TH, 'general_eval_metrics.csv'))
        current_mcc = list(result['mcc'])[0]
        if biggest_mcc >= current_mcc:
            count += 1
        else:
            biggest_mcc = current_mcc
            mt = t
            count = 0
        print(mt, biggest_mcc)
        if count == 2 and not args.allth:
            break
