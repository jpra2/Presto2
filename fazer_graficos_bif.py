
import numpy as np
import matplotlib.pyplot as plt
import os

caminho2 = '/home/joao/run/back2_multiescala/Presto2/simulacoes/bifasico'
caminho1 = '/home/joao/run/back2_sol_direta/Presto2/simulacoes/bifasico'
arquivo1 = 'fluxo_multiescala_bif'
arquivo2 = 'fluxo_malha_fina_bif'
arquivo3 = 'prod_'



f1 = 'soma_prod'
t = 'tempo'
po = 'prod_o'
pw = 'prod_w'

os.chdir(caminho2)

prod_o = []
prod_o_ms = []
tempo = []
tempo_ms = []
prod_w_ms = []
prod_w = []
loops = 194
continuar = True
cont = 0

# Multiescala

while(cont <= loops):
    arquivo = arquivo3 + str(cont) + '.txt'

    with open(arquivo, 'r') as arq:
        text = arq.readlines()
    for j in text:
        if j[0:len(po)] == po:
            prod_o_ms.append(float(j.split(':')[1][0:-1]))
            #prod_o_ms.insert(-1, float(j.split(':')[1][0:-1]))
        elif j[0:len(t)] == t:
            tempo_ms.append(float(j.split(':')[1][0:-1]))
            #tempo_ms.insert(-1, float(j.split(':')[1][0:-1]))
        elif j[0:len(pw)] == pw:
            prod_w_ms.append(float(j.split(':')[1][0:-1]))
            #prod_w_ms.insert(-1, float(j.split(':')[1][0:-1]))
        else:
            pass

    cont += 1



prod_w_ms = np.array(prod_w_ms)
prod_o_ms = np.array(prod_o_ms)
wor_ms = prod_w_ms/prod_o_ms
tempo_ms = np.array(tempo_ms)

prod_o_ms_2 = np.zeros(len(prod_o_ms))
for i in range(len(prod_o_ms)):
    if i == 0:
        prod_o_ms_2[i] = prod_o_ms[i]
    else:
        prod_o_ms_2[i] = prod_o_ms[i] + prod_o_ms_2[i-1]
    # print(prod_o_ms[i])
    # print(prod_o_ms_2[i])




#######################################################
# Solucao direta

os.chdir(caminho1)
continuar = True
cont = 0

while(cont <= loops):
    arquivo = arquivo3 + str(cont) + '.txt'

    with open(arquivo, 'r') as arq:
        text = arq.readlines()
    for j in text:
        if j[0:len(po)] == po:
            prod_o.append(float(j.split(':')[1][0:-1]))
            #prod_o_ms.insert(-1, float(j.split(':')[1][0:-1]))
        elif j[0:len(t)] == t:
            tempo.append(float(j.split(':')[1][0:-1]))
            #tempo_ms.insert(-1, float(j.split(':')[1][0:-1]))
        elif j[0:len(pw)] == pw:
            prod_w.append(float(j.split(':')[1][0:-1]))
            #prod_w_ms.insert(-1, float(j.split(':')[1][0:-1]))
        else:
            pass

    cont += 1



prod_w = np.array(prod_w)
prod_o = np.array(prod_o)
wor = prod_w/prod_o
tempo = np.array(tempo)

prod_o_2 = np.zeros(len(prod_o))
for i in range(len(prod_o)):
    if i == 0:
        prod_o_2[i] = prod_o[i]
    else:
        prod_o_2[i] = prod_o[i] + prod_o_2[i-1]
    # print(prod_o_ms[i])
    # print(prod_o_ms_2[i])


# print(tempo_ms)
# print(prod_o_ms)
#
# print(len(tempo_ms))
# print(len(prod_o_ms))

# os.chdir('/home/joao/Dropbox')

caminho_imagens = '/home/joao/Imagens/imagens_producao'
os.chdir(caminho_imagens)

plt.figure(1)
plt.plot(tempo_ms, prod_o_ms_2, 'r', linewidth = 2.0, label = 'Multiescala')
plt.plot(tempo, prod_o_2, 'k', linewidth = 1.0, label = 'Solucao direta')
#plt.xlabel('Tempo')
#plt.ylabel('Produção de óleo')
# print(max([max(tempo), max(tempo_ms)]))
# print(min([min(prod_o_2), min(prod_o)]))
# print()

#import pdb; pdb.set_trace()
plt.title('Produção x Tempo')
plt.axis([0, max([max(tempo), max(tempo_ms)]), min([min(prod_o_2), min(prod_o)]), max([max(prod_o_2), max(prod_o)])])
plt.legend(loc = 'best', shadow = True)
plt.savefig('producao.png')
plt.show()

plt.figure(2)
plt.plot(tempo_ms, wor_ms, 'r', linewidth = 2.0, label = 'Multiescala')
plt.plot(tempo, wor, 'k', linewidth = 1.0, label = 'Solucao direta')
#plt.xlabel('Tempo')
#plt.ylabel('Produção de óleo')
plt.title('Wor x Tempo')
plt.axis([0, max([max(tempo), max(tempo_ms)]), min([min(wor_ms), min(wor)]), max([max(wor_ms), max(wor_ms)])])
plt.legend(loc = 'best', shadow = True)
plt.savefig('wor.png')
plt.show()
