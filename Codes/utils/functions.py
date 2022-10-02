import numpy as np

def in_FoV(lower_limit, upper_limit, *p):
    result = True
    for i in p:
        if np.sum(i<lower_limit) > 0 or np.sum(i>=upper_limit) > 0:
            result = False
            break
    return result
