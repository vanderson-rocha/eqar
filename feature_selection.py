import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import sys
from utils import *
from constants import *
import argparse
from sklearn.feature_selection import VarianceThreshold

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        print(message)
        sys.exit(2)

def parse_args(argv):
    parser = DefaultHelpParser(formatter_class = argparse.RawTextHelpFormatter)
    parser._optionals.title = 'Show Help'
    parser_dataset_group = parser.add_argument_group('Dataset Parameters')
    parser_dataset_group.add_argument(
        '-d', '--dataset', metavar = 'DATASET',
        help = 'Dataset (csv file)', type = str, required = True)
    parser_dataset_group.add_argument(
        '--sep', metavar = 'SEPARATOR', type = str, default = ',',
        help = 'Dataset Features Separator. Default: , (comma)')
    parser_dataset_group.add_argument(
        '--class-column', type = str,
        default = 'class', metavar = 'CLASS_COLUMN',
        help = 'Class Column ID. Default: class')
    parser_additional_group = parser.add_argument_group('Additional Parameters')
    parser_additional_group.add_argument(
        '-fs', '--feature-selection', metavar = 'METRIC',
        choices = fs_metrics,
        help = 'Metric For Feature Selection. Choices: ' + str(fs_metrics),
        type = str.lower, required = True)
    args = parser.parse_args(argv)
    return args

# >>>> Correlation
def calculate_correlation(features):
    correlation = features.corr(method='kendall')
    feature_count = list()
    for index in correlation.index:
        count = 0
        for column in correlation.columns:
            if index != column and abs(correlation.loc[index, column]) > 0.2:
                #print(f'{index} -> {column}: {abs(correlation.loc[index, column]):.2f}')
                count += 1
        feature_count.append([index, count])
    df = pd.DataFrame(feature_count, columns = ['feature', 'count'])
    df.sort_values(by = ['count'], ascending = False, inplace = True)
    return df

def correlation_features(dataset, args, class_id = None):
    if class_id == None:
        ds = dataset
    else:
        ds = dataset[(dataset[args.class_column] == class_id)]
    X_dataset, y_dataset = get_X_y(args, ds)
    corr = calculate_correlation(X_dataset)
    th = corr['count'].max() * 0.2
    corr = corr[corr['count'] >= th]
    selected_features = list(corr['feature'])
    return selected_features

def correlation_dataset(dataset, args):
    global class_distribution
    global columns_list
    if args.feature_selection == 'corr':
        selected_features = correlation_features(dataset, args)
        print('corr', len(selected_features))
        selected_features.append(args.class_column)
        ds = dataset[selected_features]
    elif args.feature_selection == 'scorr':
        selected_features = list()
        for class_id in class_distribution.index:
            class_features = correlation_features(dataset, args, class_id)
            print('scorr', class_id, len(class_features))
            selected_features = list(set(selected_features) | set(class_features))
        selected_features.append(args.class_column)
        ds = dataset[selected_features]
    elif args.feature_selection == 'xcorr':
        ds = dataset
        for class_id in class_distribution.index:
            class_features = correlation_features(dataset, args, class_id)
            print('xcorr', class_id, len(class_features))
            to_zero = list(set(columns_list) - set(class_features))
            #print(to_zero)
            for ft in to_zero:
                ds.loc[(ds[ft] != 0) & (ds[args.class_column] == class_id), ft] = 0
    return ds

# >>>> PRNR - SigPID
def S_B(j):
    global prnr_benign_dataset
    global prnr_malware_dataset
    sigmaBij = prnr_benign_dataset[j].sum()
    sizeBj = prnr_benign_dataset.shape[0]
    sizeMj = prnr_malware_dataset.shape[0]
    return (sigmaBij / sizeBj) * sizeMj

def PRNR(j):
    global prnr_malware_dataset
    sigmaMij = prnr_malware_dataset[j].sum()
    S_Bj = S_B(j)
    a = sigmaMij - S_Bj
    b = sigmaMij + S_Bj
    b = b if b != 0.0 else 0.001
    r = a/b
    return r

def calculate_prnr(dataset, args):
    global prnr_benign_dataset
    global prnr_malware_dataset
    ds = drop_internet(dataset)
    prnr_benign_dataset = ds[(ds[args.class_column] == 0)]
    prnr_malware_dataset = ds[(ds[args.class_column] == 1)]

    prnr_list = list()
    features = ds.columns.values.tolist()
    features.remove(args.class_column)
    for feature in features:
        rate = PRNR(feature)
        prnr_list.append([feature, rate])
    df = pd.DataFrame(prnr_list, columns = ['feature', 'rate'])
    df.sort_values(by = ['rate'], ascending = False, inplace = True)
    return df

def prnr_features(dataset, prnr, class_id = None):
    if class_id == None:
        prnr_df = prnr[(prnr['rate'] <= -0.2) | (prnr['rate'] >= 0.2)]
    elif class_id == 0:
        prnr_df = prnr[prnr['rate'] <= -0.2]
    elif class_id == 1:
        prnr_df = prnr[prnr['rate'] >= 0.2]
    selected_features = list(prnr_df['feature'])
    return selected_features

def prnr_dataset(dataset, args):
    global class_distribution
    global columns_list
    prnr = calculate_prnr(dataset, args)
    if args.feature_selection == 'prnr':
        selected_features = prnr_features(dataset, prnr)
        print('prnr', len(selected_features))
        selected_features.append(args.class_column)
        ds = dataset[selected_features]
    elif args.feature_selection == 'xprnr':
        ds = dataset
        for class_id in class_distribution.index:
            class_features = prnr_features(dataset, prnr, class_id)
            print('xprnr', class_id, len(class_features))
            to_zero = list(set(columns_list) - set(class_features))
            for ft in to_zero:
                ds.loc[(ds[ft] != 0) & (ds[args.class_column] == class_id), ft] = 0

    return ds

# >>>> Feature Discrimination
def fib(feature):
    global fd_benign_dataset
    return len(fd_benign_dataset[fd_benign_dataset[feature] == 1]) / len(fd_benign_dataset)

def fim(feature):
    global fd_malware_dataset
    return len(fd_malware_dataset[fd_malware_dataset[feature] == 1]) / len(fd_malware_dataset)

def calculate_fd(dataset, args):
    global fd_benign_dataset
    global fd_malware_dataset
    fd_benign_dataset = dataset[(dataset[args.class_column] == 0)]
    fd_malware_dataset = dataset[(dataset[args.class_column] == 1)]

    fd_list = list()
    features = dataset.drop(columns=[args.class_column])
    for ft in features:
        fb = fib(ft)
        fb = fb if fb != 0.0 else 0.001
        fm = fim(ft)
        fm = fm if fm != 0.0 else 0.001
        score = 1.0 - (min(fb, fm)/max(fb, fm))
        fd_list.append([ft,score])
    df = pd.DataFrame(fd_list, columns = ['feature', 'score'])
    df.sort_values(by = ['score'], ascending = False, inplace = True)
    return df

def fd_dataset(dataset, args):
    fd = calculate_fd(dataset, args)
    th = fd['score'].max() * 0.2
    fd = fd[fd['score'] >= th]
    selected_features = list(fd['feature'])
    print('fd', len(selected_features))
    selected_features.append(args.class_column)
    df = dataset[selected_features]
    return df

# >>>> Information Gain
def calculate_ig(features, target):
    features_names = features.columns
    ig = mutual_info_classif(features, target, random_state = 0)
    data = {'feature': features_names, 'score': ig}
    df = pd.DataFrame(data)
    df.sort_values(by=['score'], ascending=False, inplace = True)
    return df

def ig_dataset(dataset, args):
    X_dataset, y_dataset = get_X_y(args, dataset)
    ig = calculate_ig(X_dataset, y_dataset)
    th = 0.0 #ig['score'].max() * 0.2
    ig = ig[ig['score'] > th]
    selected_features = list(ig['feature'])
    print('ig', len(selected_features))
    selected_features.append(args.class_column)
    df = dataset[selected_features]
    return df

def var_features(dataset, args, class_id = None):
    if class_id == None:
        ds = dataset
    else:
        ds = dataset[(dataset[args.class_column] == class_id)]
    X_dataset, y_dataset = get_X_y(args, ds)
    #var = VarianceThreshold(threshold = (0.2 * (1.0 - 0.2)))
    var = VarianceThreshold(threshold = 0.2)
    X_high_variance = var.fit_transform(X_dataset)
    selected_features = list(var.get_feature_names_out(X_dataset.columns.tolist()))
    return selected_features

def var_dataset(dataset, args):
    global class_distribution
    global columns_list
    if args.feature_selection == 'var':
        selected_features = var_features(dataset, args)
        print('var', len(selected_features))
        selected_features.append(args.class_column)
        ds = dataset[selected_features]
    elif args.feature_selection == 'svar':
        selected_features = list()
        for class_id in class_distribution.index:
            class_features = var_features(dataset, args, class_id)
            print('svar', class_id, len(class_features))
            selected_features = list(set(selected_features) | set(class_features))
        selected_features.append(args.class_column)
        ds = dataset[selected_features]
    elif args.feature_selection == 'xvar':
        ds = dataset
        for class_id in class_distribution.index:
            class_features = var_features(dataset, args, class_id)
            print('xvar', class_id, len(class_features))
            to_zero = list(set(columns_list) - set(class_features))
            for ft in to_zero:
                ds.loc[(ds[ft] != 0) & (ds[args.class_column] == class_id), ft] = 0
    return ds

if __name__=="__main__":
    global class_distribution
    global columns_list
    args = parse_args(sys.argv[1:])
    try:
        dataset = pd.read_csv(args.dataset, sep = args.sep)
    except BaseException as e:
        print(e)
        exit(1)

    filename = (args.dataset).split('/')[-1]
    filename = f'{args.feature_selection}_{filename}'

    X_dataset, y_dataset = get_X_y(args, dataset)

    class_distribution = y_dataset.value_counts()
    class_distribution.sort_index(inplace = True)
    columns_list = dataset.columns.values.tolist()
    columns_list.remove(args.class_column)

    if args.feature_selection == 'ig':
        dataset = ig_dataset(dataset, args)
    elif args.feature_selection == 'fd':
        dataset = fd_dataset(dataset, args)
    elif args.feature_selection.endswith('corr'):
        dataset = correlation_dataset(dataset, args)
    elif args.feature_selection.endswith('prnr'):
        dataset = prnr_dataset(dataset, args)
    elif args.feature_selection.endswith('var'):
        dataset = var_dataset(dataset, args)

    if args.feature_selection.startswith('x'):
        features_sum = dataset.sum()
        selected_features = list()
        for feature, sum in features_sum.items():
            if sum > 0:
                selected_features.append(feature)
        dataset = dataset[selected_features]

    print(dataset.shape)
    dataset.to_csv(filename, index = False)
