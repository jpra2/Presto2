import numpy as np
from test34 import MsClassic_mono
import time
# import cProfile as cp

t1 = time.time()
# pr = cp.Profile()
# pr.enable()
sim_mono = MsClassic_mono(ind = True)
# sim_mono = MsClassic_mono()
# pr.disable()
# pr.print_stats(sort = 'time')



t2 = time.time()
print('\n')
print('tempo de simulacao')
print(t2-t1)
