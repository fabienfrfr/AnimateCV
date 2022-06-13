# fabienfrfr - 20220613

import numpy as np

def PrimeFactors(n):
    LIST = []
    # pairs
    while n % 2 == 0:
        LIST += [2]
        n = n / 2
    # odd
    for i in range(3,int(np.sqrt(n))+1,2):
        while n % i == 0:
            LIST += [i]
            n = n / i
    # is prime
    if n > 2: LIST += [n]
    return np.array(LIST, dtype=int)

def GRID_GEN(NB_TOT) :
    # Grid generator
    GRID = np.arange(NB_TOT)
    RESH = PrimeFactors(NB_TOT)
    if RESH.size == 1 :
        RESH = [1] + list(RESH)
        GRID = GRID[None]
    elif RESH.size == 2 :
        GRID = GRID.reshape(RESH)
    elif RESH.size == 3 :
        RESH = (np.product(RESH[:2]),RESH[-1])
        GRID = GRID.reshape(RESH)
    else :
        CUT = np.ceil(RESH.size/2).astype(int)
        RESH = (np.product(RESH[:CUT]),np.product(RESH[CUT:]))
        GRID = GRID.reshape(RESH)
    RESH = tuple(RESH)
    return (GRID,RESH)