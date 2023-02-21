import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from functools import partial
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from fim import apriori, eclat, fpgrowth
from termcolor import colored, cprint
import shutil
import os
import logging
from imblearn.under_sampling import NearMiss, TomekLinks
from imblearn.over_sampling import SMOTE

def parallelize_func(func, fixed_parameter, const_parameters = None, cores = cpu_count()):
    logger = logging.getLogger('EQAR')
    tqdm_disable = logger.getEffectiveLevel() > logging.INFO
    x = len(fixed_parameter)
    with Pool(cores) as pool:
        if const_parameters == None:
            result = list(tqdm(pool.imap(func, fixed_parameter), total = x, disable = tqdm_disable))
        else:
            result = list(tqdm(pool.imap(partial(func, const_parameters = const_parameters), fixed_parameter), total=x, disable = tqdm_disable))
    return result

def file_content(location, fold, type_file):
    f_name = os.path.join(location, f'{fold}_{type_file}')
    ct = None
    try:
        with open(f_name) as f:
            lines = f.readlines()
            if type_file == 'log':
                ct = lines[0]
            elif type_file == 'prediction':
                ct = [int(l[:-1]) for l in lines]
        return ct
    except BaseException as e:
        return ct

def update_log(location, fold, step):
    f_name = os.path.join(location, f'{fold}_log')
    with open(f_name, 'w') as f:
        f.write(step)
        f.close()

def load_runtimes(location, fold):
    f_name = os.path.join(location, f'{fold}_runtimes.csv')
    runtimes = pd.read_csv(f_name)
    return runtimes

def save_runtimes(location, runtimes, fold):
    f_name = os.path.join(location, f'{fold}_runtimes.csv')
    runtimes.to_csv(f_name, index = False)

def load_prob_prediction(location, fold):
    f_name = os.path.join(location, f'{fold}_prob_prediction.csv')
    prob_prediction = pd.read_csv(f_name)
    prob_prediction.columns = prob_prediction.columns.astype(int) #check to multiclass
    return prob_prediction

def save_prob_prediction(location, prob_prediction, fold):
    f_name = os.path.join(location, f'{fold}_prob_prediction.csv')
    prob_prediction.to_csv(f_name, index = False)

def save_prediction(location, prediction, fold):
    f_name = os.path.join(location, f'{fold}_prediction')
    with open(f_name, 'w') as f:
        for pred in prediction:
            # write each prediction on a new line
            f.write(f'{pred}\n')
        f.close()

def check_directories(dataset_file, args):
    # - ->> directories to save data <<- -#
    root_path = os.path.dirname(os.path.realpath(__file__))
    prefix = ''
    if args.balance_dataset:
        prefix += f'bl_{args.balance_dataset}_'
    if args.use_proportional_support:
        prefix += 'ps_'

    dir_name = (dataset_file).split('/')[-1]
    dir_name = prefix + dir_name.split('.')[0]
    dir_name += f'_S{int(args.min_support * 100.0)}'
    dir_name += f'_C{int(args.min_confidence * 100.0)}'
    dir_name += f'_L{args.max_length}'

    path = os.path.join(root_path, 'run_files', dir_name)
    DIR_BASE = path

    if args.qualify:
        path = os.path.join(path, args.qualify)
    else:
        path = os.path.join(path, 'supp')
    DIR_QFY = path

    th_dir = f'T{int(args.threshold * 100.0)}'
    DIR_TH = os.path.join(path, th_dir)

    if args.overwrite:
        overwrite_dir = locals()['DIR_' + (args.overwrite).upper()]
        if os.path.exists(overwrite_dir):
            x = input('Confirm Overwrite? [y/n]: ')
            if x in ['y', 'Y']:
                shutil.rmtree(overwrite_dir)

    if not os.path.exists(DIR_TH):
        os.makedirs(DIR_TH)

    return DIR_BASE, DIR_QFY, DIR_TH

#Dictionary For Column Names
report_colnames = {
    'a': 'support_itemset_absolute',
    's': 'support_itemset_relative',
    'b': 'support_antecedent_absolute',
    'x': 'support_antecedent_relative',
    'X': 'support_antecedent_relative_pct',
    'h': 'support_consequent_absolute',
    'y': 'support_consequent_relative',
    'Y': 'support_consequent_relative_pct',
    'c': 'confidence',
    'C': 'confidence_pct',
    'l': 'lift',
    'L': 'lift_pct',
    'e': 'evaluation',
    'E': 'evaluation_pct',
    'S': 'support_emptyset'
}

def execute_fim(dataset, args, report, factor):
    support = args.min_support * 100.0 * factor
    confidence = args.min_confidence * 100.0
    algorithm_fim = globals()[args.algorithm]
    rules_fim = algorithm_fim(dataset, target = 'r', zmin = 3, zmax = args.max_length,
                    supp = support, conf = confidence, report = report, mode = 'o')
    return rules_fim

def dataset_transaction(transaction):
    num_columns = len(transaction)
    t = list()
    for i in range(0, num_columns):
        if transaction[i]:
            t.append(i)
    #if len(t) > 2:
    return t

def to_fim_format(dataset, full = False):
    data = dataset.values.tolist()
    result = parallelize_func(dataset_transaction, data)
    #result = list(filter(None, result))
    return result

def to_pandas_dataframe(data, report):
    colnames = ['consequent', 'antecedent'] + [report_colnames.get(r, r) for r in list(report)]
    dataset_df = pd.DataFrame(data, columns = colnames)
    s = dataset_df[['consequent', 'antecedent']].values.tolist()
    size_list = [len(i[1]) for i in s]
    dataset_df['size'] = size_list
    dataset_df.rename(columns = {'support_itemset_relative':'support'}, inplace = True)
    return dataset_df

def eval_metrics_dataframe(classification, prediction, num_rules = -1):
    tn, fp, fn, tp = confusion_matrix(classification, prediction).ravel()
    accuracy = metrics.accuracy_score(classification, prediction)
    precision = metrics.precision_score(classification, prediction, zero_division = 0)
    recall = metrics.recall_score(classification, prediction, zero_division = 0)
    f1_score = metrics.f1_score(classification, prediction, zero_division = 0)
    mcc = metrics.matthews_corrcoef(classification, prediction)
    roc_auc = metrics.roc_auc_score(classification, prediction)
    eval_metrics_dict = {
        'num_rules': [num_rules],
        'tp': [tp],
        'tn': [tn],
        'fp': [fp],
        'fn': [fn],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1_score],
        'mcc': [mcc],
        'roc_auc': [roc_auc]
    }
    eval_metrics = pd.DataFrame(eval_metrics_dict)
    return eval_metrics

def get_X_y(args, dataset):
    logger = logging.getLogger('EQAR')
    if(args.class_column not in dataset.columns):
        logger.exception(f'Class Column {args.class_column} Not Found in Dataset {args.dataset}')
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]
    return X, y

def balance_dataset(args, dataset):
    X, y = get_X_y(args, dataset)
    method = args.balance_dataset
    if method == 'smote':
        mth = SMOTE(random_state = 0)
    elif method == 'nearmiss':
        mth = NearMiss()
    X_res, y_res = mth.fit_resample(X, y)
    balanced_dataset = pd.concat([X_res, y_res], axis = 1)
    return balanced_dataset

def drop_internet(dataset):
    cols = []
    to_drop = ['android.permission.INTERNET', 'INTERNET']
    features = dataset.columns.values.tolist()
    cols = list(set(features).intersection(to_drop))
    ds = dataset.drop(columns=cols)
    return ds

def correlation_dataset(dataset, args, class_id = None):
    if class_id == None:
        ds = dataset
    else:
        ds = dataset[(dataset[args.class_column] == class_id)]
    X_dataset, y_dataset = get_X_y(args, ds)
    correlation = X_dataset.corr(method='kendall')
    index_count = list()

    for index in correlation.index:
        count = 0
        for column in correlation.columns:
            if index != column and correlation.loc[index, column] > 0.2:
                count += 1
        index_count.append([index, count])

    df = pd.DataFrame(index_count, columns = ['feature', 'count'])
    th = df['count'].max() * 0.2
    df = df[df['count'] < th]
    to_zero = list(df['feature'])
    for ft in to_zero:
        X_dataset.loc[X_dataset[ft] > 0, ft] = 0

    X_dataset[args.class_column] = y_dataset
    return X_dataset
