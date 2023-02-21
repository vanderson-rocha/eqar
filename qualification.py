def q_accuracy(p, n, P, N):
    return (p + (N - n)) / (P + N)

def q_coverage(p, n, P, N):
    return (p + n) / (P + N)

def q_precision(p, n, P, N):
    return p / (p + n)

def q_logical_sufficiency(p, n, P, N):
    a = p * N
    b = n * P
    if b == 0:
        b = 0.0001
    return a / b

def q_bayesian_confirmation(p, n, P, N):
    a = p / (p + N)
    b = P - p
    c = P -p + N - n
    return a - (b / c)

def q_kappa(p, n, P, N):
    a = (p * N) - (P * n)
    b = (P + N) * (p + n + P)
    c = 2 * (p + n) * P
    return 2 * (a / (b - c))

def q_zhang(p, n, P, N):
    a = (p * N) - (P * n)
    b = p * N
    c = P * n
    return a / max([b, c])

def q_correlation(p, n, P, N):
    import math
    a = (p * N) - (P * n)
    b = P * N * (p + n)
    c = P -p + N - n
    r = math.sqrt(b * c)
    return a / r

def q_wlaplace(p, n, P, N):
    a = (p + 1) * (P + N)
    b = (p + n + 2) * P
    return a / b

def coleman(p, n, P, N):
    a = ((P + N) * (p / (p + n))) - P
    return a / N

def cohen(p, n, P, N):
    a = ((P + N) * (p / (p + n))) - P
    b = (P + N) / 2
    c = (p + n + P) / (p + n)
    return a / ((b * c) - P)

def q_c1(p, n, P, N):
    a = coleman(p, n, P, N)
    b = cohen(p, n, P, N)
    return a * ((2 + b) / 3)

def q_c2(p, n, P, N):
    a = coleman(p, n, P, N)
    b = 1 + (p / P)
    return a * 0.5 * b
