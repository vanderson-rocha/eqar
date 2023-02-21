from qualification import *

#Dictionary For Rules Qualify Functions
rules_quality_functions = {
    'acc': q_accuracy,
    'cov': q_coverage,
    'prec': q_precision,
    'ls': q_logical_sufficiency,
    'bc': q_bayesian_confirmation,
    'kap': q_kappa,
    'zha': q_zhang,
    'corr': q_correlation,
    'c1': q_c1,
    'c2': q_c2,
    'wl': q_wlaplace,
}

step_dict = {
    'generate_rules': 0,
    'filter_rules': 1,
    'quality_parameters': 2,
    'qualify_rules': 3,
    'best_rules': 4,
    'test_apps': 5,
    'results_calc': 6,
    'finished': 7
}

quality_metrics = ['acc', 'c1', 'c2', 'bc', 'kap', 'zha', 'wl', 'corr', 'cov', 'prec', 'ls']

directories_id = ['base', 'qfy', 'th']

fs_metrics = ['ig', 'fd', 'corr', 'scorr', 'xcorr', 'prnr', 'xprnr', 'var', 'svar', 'xvar']
