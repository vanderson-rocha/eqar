import logging
from utils import *
import pandas as pd
import ast
import numpy as np
from itertools import product

def save_rules(location, rules, fold, type = 'class_rules'):
    if type == 'class_rules':
        class_index = rules.index
        for c in class_index:
            f_name = os.path.join(location, f'{fold}_class_{c}_rules.csv')
            r = rules[c]
            r.to_csv(f_name, index = False)
    else:
        f_name = os.path.join(location, f'{fold}_rules.csv')
        rules.to_csv(f_name, index = False)

def load_rules(location, fold, class_index = list()):
    if len(class_index):
        data = list()
        for c in class_index:
            f_name = os.path.join(location, f'{fold}_class_{c}_rules.csv')
            class_rules = pd.read_csv(f_name)
            class_rules['rule'] = class_rules['rule'].apply(ast.literal_eval)
            data.append(class_rules)
        rules = pd.Series(data, index = class_index)
    else:
        f_name = os.path.join(location, f'{fold}_rules.csv')
        rules = pd.read_csv(f_name)
        rules['rule'] = rules['rule'].apply(ast.literal_eval)

    return rules
'''
def rules_difference(list_a, list_b):
    a = set(map(tuple, list_a))
    b = set(map(tuple, list_b))
    list_d = a - b
    ordered = sorted(list(list_d), key = len)
    return ordered
'''
def rules_intersection(list_a, list_b):
    a = set(map(tuple, list_a))
    b = set(map(tuple, list_b))
    list_i = a & b
    return list(list_i)

def generate_rules(dataset_fim, args, index, factor):
    report = 's'
    rules_fim = execute_fim(dataset_fim, args, report, factor)
    rules_dataframe = to_pandas_dataframe(rules_fim, report)
    rules = rules_dataframe[rules_dataframe['consequent'] == index]
    rules = rules.drop('consequent', axis = 1)
    rules.rename(columns = {'antecedent':'rule'}, inplace = True)
    rules['rule'] = [tuple(sorted(x)) for x in rules['rule']]
    return rules

def target_rules(rules, class_id, args):
    logger = logging.getLogger('EQAR')
    supp = args.min_support
    if len(rules) <= args.target_rules:
        logger.info(f'[Class {class_id}] Target Number of Rules Satisfied. Support: {int(supp * 100.00)}%')
        return rules
    support_step = 0.01
    flag = True
    target_rules = pd.DataFrame()
    while flag:
        supp += support_step
        target_rules = rules[rules['support'] >= supp]
        flag = len(target_rules) > args.target_rules
    logger.info(f'[Class {class_id}] Target Number of Rules Satisfied. Support: {int(supp * 100.00)}%')
    return target_rules

def remove_rules_intersection(rules, class_distribution, args):
    logger = logging.getLogger('EQAR')
    data = list()
    for class_id in class_distribution.index:
        r = rules[class_id]
        rules_list = list(r['rule'])
        data.append(rules_list)
    r = pd.Series(data, index = class_distribution.index)

    filtered_rules = rules.copy()
    for x, y in product(class_distribution.index, class_distribution.index):
        if y > x:
            logger.info(f'Processing Intersection: Class {x} and Class {y}')
            rules_i = rules_intersection(r[x], r[y]) #check for multiclass
            drop_in_x = rules_i
            drop_in_y = rules_i
            lrx = filtered_rules[x]
            lry = filtered_rules[y]

            to_drop_index = lrx[lrx['rule'].isin(drop_in_x)].index
            lrx = lrx.drop(to_drop_index, axis = 0)
            filtered_rules[x] = lrx
            logger.info(f'[Class {x}] {len(lrx)} Association Rules')

            to_drop_index = lry[lry['rule'].isin(drop_in_y)].index
            lry = lry.drop(to_drop_index, axis = 0)
            filtered_rules[y] = lry
            logger.info(f'[Class {y}] {len(lry)} Association Rules')

    return filtered_rules

def rules_superset(rule_to_check, const_parameters):
    rules_list = const_parameters[0]
    if rule_to_check in rules_list:
        rules_list.remove(rule_to_check)
    args = const_parameters[1]
    max_len = args.max_length
    rule_len = len(rule_to_check)
    if rule_len == 2:
        return None
    minors_rules = [r for r in rules_list if len(r) < rule_len and rule_to_check[0] <= r[0] and rule_to_check[-1] >= r[-1]]
    is_superset = any(set(rule_to_check).issuperset(r) for r in minors_rules)
    if is_superset:
        return rule_to_check
    return None
'''
def rules_superset(rule_to_check, const_parameters):
    rules_list = const_parameters[0]
    rules_list.sort(key=len)
    if rule_to_check in rules_list:
        rules_list.remove(rule_to_check)
    args = const_parameters[1]
    max_len = args.max_length
    rule_len = len(rule_to_check)
    if rule_len == 2:
        return None
    minors_rules = [r for r in rules_list if len(r) < rule_len and rule_to_check[0] <= r[0] and rule_to_check[-1] >= r[-1]]
    minors_rules = [r for r in minors_rules if len(r) <= rule_len - 2] # Filter further by length
    for r in minors_rules:
        if all(c in rule_to_check for c in r):
            return rule_to_check
    return None
'''

def remove_rules_supersets(rules, class_distribution, args):
    logger = logging.getLogger('EQAR')
    filtered_rules = rules.copy()
    for class_id in class_distribution.index:
        #if class_id == 0:
            #continue
        logger.info(f'[Class {class_id}] Removing Superset Rules')
        r = rules[class_id]
        rules_list = list(r['rule'])
        rules_s = parallelize_func(rules_superset, rules_list, const_parameters = [rules_list, args]) #check to multclass
        rules_s = list(filter(None, rules_s))
        lrx = filtered_rules[class_id]
        to_drop_index = lrx[lrx['rule'].isin(rules_s)].index
        lrx = lrx.drop(to_drop_index, axis = 0)
        filtered_rules[class_id] = lrx
        logger.info(f'[Class {class_id}] {len(lrx)} Association Rules')
    return filtered_rules
