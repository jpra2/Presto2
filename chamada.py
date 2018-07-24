import os

principal = '/home/joao/git/back2/Presto2'

chamada1 = 'sudo systemctl start docker'
chamada2 = 'su -c \'setenforce 0\''
chamada4 = 'sudo docker run -v $PWD:/elliptic presto bash -c \"cd /elliptic; python -m elliptic.Preprocess structured.cfg\"'
chamada5 = 'sudo docker run -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python simulation_mono.py\"'

l2 = [chamada1, chamada2]
l1 = [chamada4, chamada5]

os.chdir(principal)

# for i in l1:
#     os.system(i)

os.system(chamada5)
