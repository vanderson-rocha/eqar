import argparse
import sys
import ast
import numpy as np
import pandas as pd
import timeit
from termcolor import colored, cprint
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from fim import apriori, eclat, fpgrowth
from spinner import Spinner
from utils import *
from qualification import *
from rules import *
from graphs import *
from constants import *
import logging
#-----------------------------------------------------------------------

def float_range(mini,maxi):
    # Define the function with default arguments
    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a Floating Point Number")
        if f <= mini or f > maxi:
            raise argparse.ArgumentTypeError("Must be > " + str(mini) + " and <= " + str(maxi))
        return f
    return float_range_checker

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        global logger
        self.print_help()
        msg = colored(message, 'red')
        logger.error(msg)
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
    parser_eqar_group = parser.add_argument_group('EQAR Parameters')
    parser_eqar_group.add_argument(
        '-a', '--algorithm', metavar = 'ALGORITHM',
        help = 'Algorithm Used to Generate Association Rules. Default: eclat',
        choices = ['apriori', 'fpgrowth', 'eclat'],
        type = str.lower, default = 'eclat')
    parser_eqar_group.add_argument(
        '-l', '--max-length', metavar = 'INT',
        help = 'Max Length of Rules (Antecedent) (must be > 2). Default: 4',
        type = int, default = 4)
    parser_eqar_group.add_argument(
        '-s', '--min-support', metavar = 'FLOAT',
        help = 'Minimum Support (must be > 0.0 and < 1.0). Default: 0.1',
        type = float_range(0.0, 1.0), default = 0.1)
    parser_eqar_group.add_argument(
        '-c', '--min-confidence', metavar = 'FLOAT',
        help = 'Minimum Confidence (must be > 0.0 and < 1.0). Default: 0.95',
        type = float_range(0.0, 1.0), default = 0.95)
    parser_eqar_group.add_argument(
        '-p', '--min-probability', metavar = 'FLOAT',
        help = 'Minimum Probability (must be > 0.0 and < 1.0). Default: 0.5',
        type = float_range(0.0, 1.0), default = 0.5)
    parser_eqar_group.add_argument(
        '-q', '--qualify', metavar = 'QUALIFY',
        choices = quality_metrics,
        help = 'Metric For Rules Qualification. Choices: ' + str(quality_metrics),
        type = str.lower, required = True)
    parser_eqar_group.add_argument(
        '-t', '--threshold', metavar = 'FLOAT',
        help = 'Percentage of Rules to be Used for Testing Samples (must be > 0.0 and < 1.0). Default: 0.2',
        type = float_range(0.0, 1.0), default = 0.2)
    parser_eqar_group.add_argument(
        '--target-rules', metavar = 'INT',
        help = 'Maximum Number of Rules To Be Qualified (in Each Fold).',
        type = int)
    parser_additional_group = parser.add_argument_group('Additional Parameters')
    parser_additional_group.add_argument(
        '--verbose', help = 'Increase Output Data.',
        action = 'store_true')
    parser_additional_group.add_argument(
        '--overwrite', metavar = 'DIRECTORY',
        choices = directories_id,
        help = 'Overwrite Directory Data. Choices: ' + str(directories_id),
        type = str.lower)
    balanced_proportional = parser_additional_group.add_mutually_exclusive_group(required = False)
    balanced_proportional.add_argument(
        '-b', '--balance-dataset', metavar = 'METHOD',
        help = 'Balance Dataset With SMOTE (Oversample) or NearMiss (Undersample).',
        choices = ['smote', 'nearmiss'],
        type = str.lower, required = False)
    balanced_proportional.add_argument(
        '-u', '--use-proportional-support', action = 'store_true',
        help = 'Use Proportional Values for Support.')
    args = parser.parse_args(argv)
    return args

def quality_parameters(rule, const_parameters):
    class_list = const_parameters[0]
    train_fim = const_parameters[1]
    classification = const_parameters[2]
    class_factor = const_parameters[3]

    rule_coverage = 0 # p + n
    p = 0
    rule_len = len(rule)
    for i in range(0, len(train_fim)):
        t = train_fim[i]
        if len(t) > rule_len and rule[0] >= t[0] and rule[-1] <= t[-1] and set(rule).issubset(t):
            rule_coverage += 1
            if class_list[i] == classification:
                p += 1
    #p = int(p * class_factor)
    n = rule_coverage - p
    return (p, n)

def qualify_rules(rule_qfy_parameters, const_parameters):
    P = const_parameters[0]
    N = const_parameters[1]
    q_function = const_parameters[2]
    factor = const_parameters[3]
    support = rule_qfy_parameters['support']
    p = rule_qfy_parameters['p']
    n = rule_qfy_parameters['n']
    q = q_function(p, n, P, N)
    return q #* support #* factor

def get_rules(dataset, args, fold):
    global DIR_BASE
    global logger
    global class_distribution
    global columns_list

    step = 'generate_rules'
    update_log(DIR_BASE, fold, step)

    class_index = columns_list.index(args.class_column)
    data = list()
    runtimes = pd.DataFrame()
    minor_class_amount = class_distribution.min()
    start_time = timeit.default_timer()

    for class_id in class_distribution.index:
        logger.info(f'[Class {class_id}] Preparing Dataset')
        class_dataset = dataset[(dataset[args.class_column] == class_id)].copy()
        class_dataset[args.class_column] = 1
        class_dataset_fim = to_fim_format(class_dataset)
        class_factor = 1.0
        if args.use_proportional_support:
            class_factor = class_distribution[class_id] / minor_class_amount
        logger.info(f'[Class {class_id}] Running FIM: Generating Association Rules')
        class_rules = generate_rules(class_dataset_fim, args, class_index, class_factor)
        logger.info(f'[Class {class_id}] Generated {len(class_rules)} Association Rules')
        data.append(class_rules)
    end_time = timeit.default_timer()
    runtime_get_rules = pd.DataFrame([[step, end_time - start_time]], columns = ['step', 'runtime'])
    runtimes = pd.concat([runtimes, runtime_get_rules])
    rules = pd.Series(data, index = class_distribution.index)
    save_rules(DIR_BASE, rules, fold)
    save_runtimes(DIR_BASE, runtimes, fold)
    return rules, runtimes

def get_filtered_rules(rules, runtimes, args, fold):
    global DIR_BASE
    global logger
    global class_distribution

    step = 'filter_rules'
    update_log(DIR_BASE, fold, step)

    logger.info(f'[Fold {fold}] Filtering Rules')
    start_time = timeit.default_timer()
    filtered_rules = remove_rules_intersection(rules, class_distribution, args)
    filtered_rules = remove_rules_supersets(filtered_rules, class_distribution, args)

    if args.target_rules:
        for class_id in class_distribution.index:
            class_filtered_rules = filtered_rules[class_id]
            filtered_rules[class_id] = target_rules(class_filtered_rules, class_id, args)

    end_time = timeit.default_timer()
    runtime_filter_rules = pd.DataFrame([[step, end_time - start_time]], columns = ['step', 'runtime'])
    runtime_filter_rules = pd.concat([runtimes, runtime_filter_rules])
    save_rules(DIR_BASE, filtered_rules, fold)
    save_runtimes(DIR_BASE, runtime_filter_rules, fold)
    return filtered_rules, runtime_filter_rules

def get_rules_quality_parameters(train, rules, runtimes, args, fold):
    global DIR_BASE
    global logger
    global class_distribution

    step = 'quality_parameters'
    update_log(DIR_BASE, fold, step)

    data = list()
    minor_class_amount = class_distribution.min()
    logger.info('Converting Train Dataset to FIM Format')
    train_list = train.drop(columns = args.class_column)
    train_fim = to_fim_format(train_list)
    start_time = timeit.default_timer()
    for class_id in class_distribution.index:
        #if class_id == 0:
            #continue
        r = rules[class_id]
        r.reset_index(drop = True, inplace = True)
        rules_list = list(r['rule'])
        logger.info(f'[Class {class_id}] Getting Parameters to Qualify Rules')
        class_factor = minor_class_amount / class_distribution[class_id]
        qfy_parameters_list = parallelize_func(quality_parameters, rules_list, const_parameters = [list(train[args.class_column]), train_fim, class_id, class_factor])
        qfy_parameters = pd.DataFrame(qfy_parameters_list, columns = ['p', 'n'])
        qfy_parameters_class_rules = pd.concat([r, qfy_parameters], axis = 1)
        data.append(qfy_parameters_class_rules)
    end_time = timeit.default_timer()
    runtime_quality_parameters = pd.DataFrame([[step, end_time - start_time]], columns = ['step', 'runtime'])
    runtime_quality_parameters = pd.concat([runtimes, runtime_quality_parameters])
    rules_qfy_parameters = pd.Series(data, index = class_distribution.index)
    save_rules(DIR_BASE, rules_qfy_parameters, fold)
    save_runtimes(DIR_BASE, runtime_quality_parameters, fold)
    return rules_qfy_parameters, runtime_quality_parameters

def get_qualified_rules(train, rules, runtimes, args, fold):
    global DIR_QFY
    global logger
    global class_distribution

    step = 'qualify_rules'
    update_log(DIR_QFY, fold, step)

    data = list()
    minor_class_amount = class_distribution.min()
    start_time = timeit.default_timer()
    for class_id in class_distribution.index:
        r = rules[class_id]
        class_factor = minor_class_amount / class_distribution[class_id]
        P = len(train[(train[args.class_column] == class_id)])
        N = len(train[(train[args.class_column] != class_id)])
        q_function = rules_quality_functions.get(args.qualify, lambda: "Invalid Qualify Measure.")
        logger.info(f'[Class {class_id}] Qualifying Rules')
        rules_iter = [row for _, row in r.iterrows()]
        rules_quality_values = parallelize_func(qualify_rules, rules_iter, const_parameters = [P, N, q_function, class_factor])
        r['q_value'] = rules_quality_values
        r = r.drop(columns = ['p', 'n'])
        data.append(r)
    end_time = timeit.default_timer()
    runtime_qualify_rules = pd.DataFrame([[step, end_time - start_time]], columns = ['step', 'runtime'])
    runtime_qualify_rules = pd.concat([runtimes, runtime_qualify_rules])
    qualified_rules = pd.Series(data, index = class_distribution.index)
    save_rules(DIR_QFY, qualified_rules, fold)
    save_runtimes(DIR_QFY, runtime_qualify_rules, fold)
    return qualified_rules, runtime_qualify_rules

def get_best_rules(rules, args, fold, attribute = 'q_value'):
    global DIR_TH
    global logger
    global class_distribution

    update_log(DIR_TH, fold, 'best_rules')
    best_rules = pd.DataFrame()
    '''
    num_rules_per_class = [len(rules[c]) for c in class_distribution.index]
    th = int(min(num_rules_per_class) * args.threshold)
    th = th if th else 1
    '''
    for c in class_distribution.index:
        r = rules[c]
        r = r.sort_values(by = [attribute], ascending = False)
        th = int(len(rules[c]) * args.threshold) + 1 # <<<<
        r = r.head(th)
        r[args.class_column] = c
        best_rules = pd.concat([best_rules, r], axis = 0)
    save_rules(DIR_TH, best_rules, fold, 'best_rules')
    return best_rules

def test_apps(test, const_parameters):
    global class_distribution
    rules = const_parameters[0]
    rules_normalization_factor = const_parameters[1]
    args = const_parameters[2]
    coverage_rules = pd.Series(0.0, index = class_distribution.index)

    for i, r in rules.iterrows():
        rule = r['rule']
        if set(rule).issubset(test):
            coverage_rules[r[args.class_column]] += 1.0

    for class_id in class_distribution.index:
        coverage_rules[class_id] *= rules_normalization_factor[class_id]

    sum = coverage_rules.sum()
    for class_id in class_distribution.index:
        if sum > 0.0:
            coverage_rules[class_id] /= sum
    return coverage_rules

def get_prediction(test, rules, runtimes, args, fold):
    global DIR_TH
    global logger
    global class_distribution
    default_class = 1 #check for multiclass

    step = 'test_apps'
    update_log(DIR_TH, fold, 'test_apps')
    logger.info(f'{len(rules)} Rules To Be Tested')
    logger.info('Converting Test Dataset to FIM Format')
    test_list = test.drop(columns = args.class_column)
    test_fim = to_fim_format(test_list)
    logger.info('Testing Applications')
    start_time = timeit.default_timer()
    rules_distribution = rules[args.class_column].value_counts()
    rules_distribution.sort_index(inplace = True)
    rules_normalization_factor = [max(rules_distribution)/rules_distribution[c] if c in rules_distribution.index else 0.0 for c in class_distribution.index]
    test_probabilistic_prediction = parallelize_func(test_apps, test_fim, const_parameters = [rules, rules_normalization_factor, args])
    end_time = timeit.default_timer()
    test_probabilistic_prediction = pd.DataFrame(test_probabilistic_prediction)
    test_probabilistic_prediction_list = test_probabilistic_prediction.values.tolist()
    #For Binary Classification
    test_prediction = [int(t[default_class] >= args.min_probability) for t in test_probabilistic_prediction_list]
    runtime_test_apps = pd.DataFrame([[step, end_time - start_time]], columns = ['step', 'runtime'])
    runtime_test_apps = pd.concat([runtimes, runtime_test_apps])
    save_prob_prediction(DIR_TH, test_probabilistic_prediction, fold)
    save_prediction(DIR_TH, test_prediction, fold)
    save_runtimes(DIR_TH, runtime_test_apps, fold)
    return test_probabilistic_prediction, test_prediction, runtime_test_apps

def get_evaluation(rules, classification, prediction):
    eval_metrics = eval_metrics_dataframe(classification, prediction, len(rules))
    return eval_metrics

def eqar(train, test, args, fold):
    global DIR_BASE
    global DIR_QFY
    global DIR_TH
    global logger
    global class_distribution
    global columns_list
    global all_malware_rules
    global sum_b
    global sum_m

    step = file_content(DIR_BASE, fold, 'log')
    stopped_step = step_dict.get(step, 0)

    if stopped_step <= step_dict.get('generate_rules'):
        logger.info(f'[Fold {fold}] Generating Rules')
        rules, runtimes = get_rules(train, args, fold)
    else:
        logger.info(f'[Fold {fold}] Rules Already Generated - Loading Data')
        rules = load_rules(DIR_BASE, fold, class_distribution.index)
        runtimes = load_runtimes(DIR_BASE, fold)

    if stopped_step <= step_dict.get('filter_rules'):
        rules, runtimes = get_filtered_rules(rules, runtimes, args, fold)

    sum_b += len(rules[0])
    sum_m += len(rules[1])

    if stopped_step <= step_dict.get('quality_parameters'):
        rules, runtimes = get_rules_quality_parameters(train, rules, runtimes, args, fold)
        update_log(DIR_BASE, fold, 'finished')

    step = file_content(DIR_QFY, fold, 'log')
    stopped_step = step_dict.get(step, 3)

    if stopped_step <= step_dict.get('qualify_rules'):
        logger.info(f'[Fold {fold}] Qualifying Rules')
        rules, runtimes = get_qualified_rules(train, rules, runtimes, args, fold)
        update_log(DIR_QFY, fold, 'finished')
    else:
        logger.info(f'[Fold {fold}] Rules Already Qualified - Loading Data')
        rules = load_rules(DIR_QFY, fold, class_distribution.index)
        runtimes = load_runtimes(DIR_QFY, fold)

    step = file_content(DIR_TH, fold, 'log')
    stopped_step = step_dict.get(step, 4)

    if stopped_step <= step_dict.get('best_rules'):
        logger.info(f'[Fold {fold}] Selecting Best Rules')
        rules = get_best_rules(rules, args, fold)
    else:
        logger.info(f'[Fold {fold}] Best Rules Already Selected - Loading Data')
        rules = load_rules(DIR_TH, fold)
        runtimes = load_runtimes(DIR_QFY, fold)

    if stopped_step <= step_dict.get('test_apps'):
        prob_prediction, prediction, runtimes = get_prediction(test, rules, runtimes, args, fold)
    else:
        logger.info(f'[Fold {fold}] Apps Already Tested - Loading Data')
        prob_prediction = load_prob_prediction(DIR_TH, fold)
        test_probabilistic_prediction_list = prob_prediction.values.tolist()
        #For Binary Classification
        default_class = 1 #check for multiclass
        prediction = [int(t[default_class] >= args.min_probability) for t in test_probabilistic_prediction_list]
        #prediction = file_content(DIR_TH, fold, 'prediction')
        runtimes = load_runtimes(DIR_TH, fold)

    classification = list(test[args.class_column])
    logger.info(f'[Fold {fold}] Calculating Evaluation Metrics')
    eval_metrics = get_evaluation(rules, classification, prediction)
    update_log(DIR_TH, fold, 'finished')
    #graph_features_convergence(DIR_TH, fold, rules, columns_list, args)
    all_malware_rules.append(rules[rules[args.class_column] == 1]['rule'])
    return eval_metrics, runtimes, prediction, prob_prediction

if __name__=="__main__":
#def run(dataset, dataset_file, args):
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    global logger
    global class_distribution
    global all_malware_rules
    global sum_b
    global sum_m

    sum_b = 0
    sum_m = 0

    logger = logging.getLogger('EQAR')
    args = parse_args(sys.argv[1:])
    if args.verbose:
        logger.setLevel(logging.INFO)

    try:
        dataset = pd.read_csv(args.dataset, sep = args.sep)
    except BaseException as e:
        msg = colored(e, 'red')
        logger.exception(msg)
        exit(1)

    global DIR_BASE
    global DIR_QFY
    global DIR_TH
    DIR_BASE, DIR_QFY, DIR_TH = check_directories(args.dataset, args)

    #dataset = drop_internet(dataset)

    if args.balance_dataset:
        dataset = balance_dataset(args, dataset)

    X_dataset, y_dataset = get_X_y(args, dataset)

    class_distribution = y_dataset.value_counts()
    class_distribution.sort_index(inplace = True)

    global columns_list
    columns_list = dataset.columns.values.tolist()

    skf = StratifiedKFold(n_splits = 5)
    fold = 1
    all_folds_eval_metrics = pd.DataFrame()
    general_prob_prediction = pd.DataFrame()
    all_malware_rules = list()
    general_class = list()
    general_prediction = list()
    for train_index, test_index in skf.split(X_dataset, y_dataset):
        train = dataset.loc[train_index,:]
        test = dataset.loc[test_index,:]
        msg = 'Executing Fold %s' % fold
        logger.info(msg)
        if not args.verbose:
            spn = Spinner(msg)
            spn.start()
        eval_metrics, runtimes, prediction, prob_prediction = eqar(train, test, args, fold)
        if not args.verbose:
            spn.stop()
        if prediction:
            #print(eval_metrics)
            all_folds_eval_metrics = all_folds_eval_metrics.append(eval_metrics, ignore_index = True)
            general_prob_prediction = general_prob_prediction.append(prob_prediction, ignore_index = True)
            general_class += list(test[args.class_column])
            general_prediction += prediction
        fold += 1
        #break

    f_name = os.path.join(DIR_TH, 'all_folds_eval_metrics.csv')
    all_folds_eval_metrics.to_csv(f_name, index = False)
    g_eval = get_evaluation(pd.DataFrame(), general_class, general_prediction)
    g_eval['num_rules'] = int(all_folds_eval_metrics['num_rules'].mean())
    f_name = os.path.join(DIR_TH, 'general_eval_metrics.csv')
    g_eval.to_csv(f_name, index = False)
    graph_features_convergence(DIR_TH, 0, all_malware_rules, columns_list, args)
    graph_roc_curve(DIR_TH, general_prob_prediction, general_class, g_eval)
    print(g_eval)
    #print(sum_b / 5.0, sum_m / 5.0)
