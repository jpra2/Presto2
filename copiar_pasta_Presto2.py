import os
import shutil

caminho9 = '/home/joao/simulacoes/999'
caminho25 = '/home/joao/simulacoes/252525'
principal = '/home/joao/git/local/Presto2'

caminho = caminho25
os.chdir(caminho)

with open('enumeracao.py', 'r') as arq:
    text = arq.readlines()
    n = int(text[-1])


with open('enumeracao.py', 'w') as arq:
    arq.write('{0}\n'.format(n+1))

pasta = 'pasta{0}'.format(n)


origem = principal
destino = os.path.join(caminho, pasta)
shutil.copytree(origem, destino)
