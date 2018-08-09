import numpy as np
import matplotlib.pyplot as plt
import os


caminho1 = '/home/joao/git/back2/Presto2/simulacoes/bifasico'
arquivo1 = 'fluxo_multiescala_bif'
arquivo2 = 'fluxo_malha_fina_bif'

arquivo = arquivo1

f1 = 'soma_prod'
t = 'tempo'

os.chdir(caminho1)

prod_o = []
prod_o_ms = []
tempo = []
tempo_ms = []

for i in os.listdir('.'):
    if i[0:len(arquivo)] == arquivo:
        with open(i, 'r') as arq:
            text = arq.readlines()
        for j in text:
            if j[0:len(f1)] == f1:
                #prod_o_ms.append(float(j.split(':')[1][0:-1]))
                prod_o_ms.insert(-1 ,float(j.split(':')[1][0:-1]))
            if j[0:len(t)] == t:
                #tempo_ms.append(float(j.split(':')[1][0:-1]))
                tempo_ms.insert(-1, float(j.split(':')[1][0:-1]))

# print(tempo_ms)
# print(prod_o_ms)
#
# print(len(tempo_ms))
# print(len(prod_o_ms))

# os.chdir('/home/joao/Dropbox')

plt.figure(1)
plt.plot(tempo_ms, prod_o_ms, 'r', label = 'Multiescala')
plt.xlabel('Tempo')
plt.ylabel('Produção de óleo')
plt.title('Produção x Tempo')
plt.axis([0, max(tempo_ms), min(prod_o_ms), max(prod_o_ms)])

plt.legend(loc = 'best', shadow = True)

# plt.savefig('producao_multiescala.png')
# plt.show()
