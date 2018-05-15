import numpy as np
from test34_bif import Msclassic_bif
import time

t1 = time.time()
sim_bif = Msclassic_bif()
sim_bif.run_2()

t2 = time.time()
print('\n')
print('tempo de simulacao')
print(t2-t1)
