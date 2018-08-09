import numpy as np
# import matplotlib.pyplot as plt
import os


caminho1 = '/home/joao/git/back2/Presto2'
arquivo = 'fluxo_multiescala.txt'
arquivo2 = 'fluxo_malha_fina.txt'

os.chdir(caminho1)

for i in os.listdir('.'):
    if i == arquivo:
        with open(i, 'r') as arq:
            text1 = arq.readlines()
    if i == arquivo2:
        with open(i, 'r') as arq:
            text2 = arq.readlines()


f1 = 'soma_prod'


for i in text1:
    if i[0:len(f1)] == f1:
        q_ms = float(i.split(':')[1][0:-1])

for i in text2:
    if i[0:len(f1)] == f1:
        q = float(i.split(':')[1][0:-1])

erro = abs(100*(q_ms - q)/q)
print(erro)
