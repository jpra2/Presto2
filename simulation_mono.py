import numpy as np
from test34 import MsClassic_mono
import time

t1 = time.time()
sim_mono = MsClassic_mono()
sim_mono.run()

t2 = time.time()
print('\n')
print('tempo de simulacao')
print(t2-t1)
