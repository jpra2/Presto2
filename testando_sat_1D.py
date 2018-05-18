import numpy as np
import matplotlib.pyplot as plt
import sys


class bifasico:
    def __init__(self, nx, ny, nz, Lx, Ly, Lz, A, k, P,
    vols_p, type_vols_p, Q, vols_q, type_vols_q, t, loops, mi_o, mi_w, fi, dim):
        self.dim = dim
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.L = Lx
        self.A = A
        self.h = L/float(nx)
        self.number_of_verts = nx + 1
        self.k = k
        self.P = P
        self.vols_p = vols_p
        self.Q = Q
        self.vols_q = vols_q
        self.create_centroids()
        self.mat_sat = np.zeros((loops, nx))
        self.mat_lamb_o = np.zeros((loops, nx))
        self.mat_lamb_w = np.zeros((loops, nx))
        self.mat_P = np.zeros((loops, nx))
        self.list_of_time = []
        self.t = t
        self.loops = loops
        self.mi_o = mi_o
        self.mi_w = mi_w
        self.loop = 0
        self.read_perm_rel()
        self.viz_x = [1, -1]
        self.n_viz_x = [1, 1]
        self.fi = fi
        self.type_vols_p = type_vols_p
        self.type_vols_q = type_vols_q
        if dim == 1:
            self.all_vols = list(range(nx))
            self.viz_x = [1, -1]
        elif dim == 2:
            self.all_vols = list(range(nx*ny))
            self.viz_x = [1, -1, nx, -nx]
        elif dim == 3:
            self.all_vols = list(range(nx*ny*nz))
            self.viz_x = [1, -1, nx, -nx, nx*ny, -nx*ny]
        else:
            print('dimensao invalida')
            sys.exit(0)

        self.Sor = 0.2
        self.Swr = 0.2
        self.V = A*self.h

    def calculate_sat(self):
        lim = 0.000001
        #import pdb; pdb.set_trace()
        for i in range(self.nx):
            #import pdb; pdb.set_trace()
            if i in self.vols_p:
                index = self.vols_p.index(i)
                if self.type_vols_p[index] == 1:
                    sat = self.mat_sat[self.loop-1, i]
                    self.mat_sat[self.loop, i] = sat
                    continue
                else:
                    pass
            div = self.div_upwind_3(i)
            fi = 0.3
            sat1 = self.mat_sat[self.loop-1, i]
            sat = sat1 + div*(self.delta_t/(fi*self.V))
            if abs(div) < lim or sat1 == (1 - self.Sor):
                self.mat_sat[self.loop, i] = sat1
                continue

            elif sat < 0:
                sat = 0
            elif sat > 1-self.Sor:
                sat = 1-self.Sor

            elif sat < 0 or sat > (1 - self.Sor):
                print('Erro: saturacao invalida')
                print('Saturacao: {0}'.format(sat))
                print('Saturacao anterior: {0}'.format(sat1))
                print('div: {0}'.format(div))
                print('gid: {0}'.format(i))
                print('fi: {0}'.format(fi))
                print('V: {0}'.format(self.V))
                print('delta_t: {0}'.format(self.delta_t))
                print('loop: {0}'.format(self.loop))

                sys.exit(0)

            self.mat_sat[self.loop, i] = sat

    def cfl_2(self, vmax, dfds):
        cfl = 0.9

        self.delta_t = (cfl*self.h)/float(vmax*dfds)

    def create_centroids(self):
        list_of_verts = []
        list_of_centroids = []
        for i in range(self.number_of_verts):
            list_of_verts.append(i*(self.h))
        for i in range(len(list_of_verts) - 1):
            v1 = list_of_verts[i+1]
            v2 = list_of_verts[i]
            centroid = (v1-v2)/2.0 + i*self.h
            list_of_centroids.append(centroid)

        self.list_of_centroids = list_of_centroids

    def div_upwind_3(self, i):

        """
        a mobilidade da interface eh dada pela media das mobilidades
        """


        q = 0.0
        pvol = self.Pf[i]
        k = self.k[i]
        lamb_w_vol = self.mat_lamb_w[self.loop-1, i]
        for j in self.viz_x:
            if i+j not in self.all_vols:
                continue
            padj = self.Pf[i+j]
            k2 = self.k[i+j]
            lamb_w_adj = self.mat_lamb_w[self.loop-1, i+j]
            keq = self.kequiv(k, k2)
            if i+j > i:
                grad_p = (padj - pvol)/float(self.h)
            else:
                grad_p = (pvol - padj)/float(self.h)
            lamb_eq = (lamb_w_vol + lamb_w_adj)/2.0
            keq = (keq*lamb_eq*self.A)
            q = q + (keq*(-grad_p))

        return q

    def kequiv(self, k1, k2):
        #keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
        keq = (2*k1*k2)/(k1+k2)

        return keq

    def pol_interp(self, S, x, y):

        """
        retorna o resultado do polinomio interpolador da saturacao usando o metodo
        das diferencas divididas, ou seja, retorna p(S)
        x = vetor da saturacao
        y = vetor que se deseja interpolar, y = f(x)
        S = saturacao
        """

        n = len(x)
        cont = 1
        est = 0
        list_delta = []

        for i in range(n-1):
            if cont == 1:
                temp = []
                for i in range(n-cont):
                    a = y[i+cont] - y[i]
                    b = x[i+cont] - x[i]
                    c = a/float(b)
                    temp.append(c)
                cont = cont+1
                list_delta.append(temp[:])
            else:
                temp = []
                for i in range(n-cont):
                    a = list_delta[est][i+1] - list_delta[est][i]
                    b = x[i+cont] - x[i]
                    c = a/float(b)
                    temp.append(c)
                cont = cont+1
                est = est+1
                list_delta.append(temp[:])

        a = []
        for i in range(n-1):
            e = list_delta[i][0]
            a.append(e)

        pol = y[0]
        mult = 1
        for i in range(n-1):
            mult = (S - x[i])*mult
            pol = pol + mult*a[i]

        return pol

    def read_perm_rel(self):
        with open("perm_rel.py", "r") as arq:
            text = arq.readlines()

        self.Sw_r = []
        self.krw_r = []
        self.kro_r = []
        self.pc_r = []

        for i in range(1, len(text)):
            a = text[i].split()
            self.Sw_r.append(float(a[0]))
            self.kro_r.append(float(a[1]))
            self.krw_r.append(float(a[2]))
            self.pc_r.append(float(a[3]))

    def set_lamb(self):

        for i in range(self.nx):
            S = self.mat_sat[self.loop, i]
            krw = self.pol_interp(S, self.Sw_r, self.krw_r)
            kro = self.pol_interp(S, self.Sw_r, self.kro_r)
            lamb_w = krw/self.mi_w
            lamb_o = kro/self.mi_o
            self.mat_lamb_w[self.loop, i] = lamb_w
            self.mat_lamb_o[self.loop, i] = lamb_o

    def set_sat_in(self):

        for i in range(self.nx):
            if i == 0:
                self.mat_sat[0, i] = 1.0
            else:
                self.mat_sat[0, i] = 0.0

    def transmissibilidade_malha_fina(self):

        A = np.zeros((self.nx, self.nx))
        b = np.zeros(self.nx)

        for i in range(self.nx):
            values = []
            ids = []
            if i in self.vols_p:
                A[i, i] = 1.0
                index = self.vols_p.index(i)
                b[i] = self.P[index]
            else:

                k = self.k[i]
                lamb_w_vol = self.mat_lamb_w[self.loop, i]
                lamb_o_vol = self.mat_lamb_o[self.loop, i]
                k = k*(lamb_w_vol + lamb_o_vol)
                for j in self.viz_x:
                    if i+j not in self.all_vols:
                        continue
                    k2 = self.k[i+j]
                    lamb_w_adj = self.mat_lamb_w[self.loop, i+j]
                    lamb_o_adj = self.mat_lamb_o[self.loop, i+j]
                    k2 = k2*(lamb_w_adj + lamb_o_adj)
                    keq = self.kequiv(k, k2)*(self.A/self.h)
                    values.append(-keq)
                    ids.append(i+j)
                values.append(-sum(values))
                ids.append(i)
                A[i, ids] = values
                if i in self.vols_q:
                    index = self.vols_q.index(i)
                    if self.type_vols_q[index] == 1:
                        b[i] += self.Q[index]
                    else:
                        b[i] += -self.Q[index]

        self.trans_fine = A
        self.b = b

    def unitary(self, l):
        uni = l/np.linalg.norm(l)
        uni = uni*uni

        return uni

    def vel_max(self):
        """
        Calcula tambem a variacao do fluxo fracionario com a saturacao
        """
        lim = 0.00001
        v2 = 0.0
        v3 = 0
        dfds2 = 0
        for i in range(self.nx):
            v = 0.0
            pvol = self.Pf[i]
            k = self.k[i]
            lamb_w_vol = self.mat_lamb_w[self.loop, i]
            lamb_o_vol = self.mat_lamb_o[self.loop, i]
            k = k*(lamb_w_vol + lamb_o_vol)
            sat_vol = self.mat_sat[self.loop, i]
            for j in self.viz_x:
                if i+j not in self.all_vols:
                    continue
                padj = self.Pf[i+j]
                k2 = self.k[i+j]
                lamb_w_adj = self.mat_lamb_w[self.loop, i+j]
                lamb_o_adj = self.mat_lamb_o[self.loop, i+j]
                k2 = k2*(lamb_w_adj + lamb_o_adj)
                sat_adj = self.mat_sat[self.loop, i+j]
                keq = self.kequiv(k, k2)/self.h
                if abs(sat_adj - sat_vol) < lim:
                    continue
                dfds = ((lamb_w_adj/(lamb_w_adj+lamb_o_adj)) - (lamb_w_vol/(lamb_w_vol+lamb_o_vol)))/float((sat_adj - sat_vol))
                v = abs(keq*(padj - pvol))
                if v > v2:
                    v2 = v
                    dfds2 = dfds
        return v2, dfds

    def run(self):
        t_ = 0
        self.set_sat_in()
        self.set_lamb()
        self.transmissibilidade_malha_fina()
        self.Pf = np.linalg.solve(self.trans_fine, self.b)
        self.mat_P[self.loop] = self.Pf
        self.list_of_time.append(t_)
        vmax, dfds = self.vel_max()
        self.cfl_2(vmax, dfds)

        fig = plt.figure()
        rect = fig.patch
        rect.set_facecolor('#31316e')
        ax1 = fig.add_subplot(2, 1, 1)
        y = self.mat_sat[self.loop]
        x = self.list_of_centroids
        ax1.plot(x, y, 'r', linestyle = '--')
        ax1.set_title('Saturacao X Tempo loop:{0} , tempo:{1}'.format(self.loop, self.list_of_time[self.loop]))
        ax1.set_xlabel('posicao')
        ax1.set_ylabel('saturacao')
        ax2 = fig.add_subplot(2, 1, 2)
        y = self.mat_P[self.loop]
        x = self.list_of_centroids
        ax2.plot(x, y, 'b', linestyle = '--')
        ax2.set_title(' Pressao X Tempo loop:{0} , tempo:{1}'.format(self.loop, self.list_of_time[self.loop]))
        ax2.set_xlabel('posicao')
        ax2.set_ylabel('pressao')
        plt.show()
        print('delta_t:{0}'.format(self.delta_t))
        print('loop:{0}'.format(self.loop))
        print('vmax:{0}'.format(vmax))
        print('\n')





        self.loop = 1
        t_ = t_ + self.delta_t
        while t_ <= self.t and self.loop <= self.loops and self.mat_sat[self.loop-1, self.nx-1] < 0.4:
            self.calculate_sat()
            self.set_lamb()
            self.transmissibilidade_malha_fina()
            self.Pf = np.linalg.solve(self.trans_fine, self.b)
            self.mat_P[self.loop] = self.Pf
            self.list_of_time.append(t_)
            vmax, dfds = self.vel_max()
            self.cfl_2(vmax, dfds)

            fig = plt.figure()
            rect = fig.patch
            rect.set_facecolor('#31312e')
            ax1 = fig.add_subplot(2, 1, 1)
            y = self.mat_sat[self.loop]
            x = self.list_of_centroids
            ax1.plot(x, y, 'r', linestyle = '--')
            ax1.set_title('Saturacao X Tempo loop:{0}'.format(self.loop))
            ax1.set_xlabel('posicao')
            ax1.set_ylabel('saturacao')
            ax2 = fig.add_subplot(2, 1, 2)
            y = self.mat_P[self.loop]
            x = self.list_of_centroids
            ax2.plot(x, y, 'b', linestyle = '--')
            ax2.set_title(' Pressao X Tempo loop:{0} '.format(self.loop))
            ax2.set_xlabel('posicao')
            ax2.set_ylabel('pressao')
            plt.show()
            plt.show()
            print('delta_t:{0}'.format(self.delta_t))
            print('loop:{0}'.format(self.loop))
            print('vmax:{0}'.format(vmax))
            print('\n')

            t_ += self.delta_t
            self.loop += 1
            self.list_of_time.append(t_)


nx = 9
ny = 1
nz = 1
Lx = 9.0
Ly = 9.0
Lz = 9.0
L = Lx
A = L**2
k = np.repeat(1.0, nx)
P = [1, 0]
vols_p = [0, 8]
type_vols_p = [1, 0]#1 = injetor, 0 = produtor
Q = []
vols_q = []
type_vols_q = []
t = 500
loops = 30
mi_o = 1.3
mi_w = 1.0
fi = 0.3
dim = 1

sim = bifasico(nx, ny, nz, Lx, Ly, Lz, A, k, P,
vols_p, type_vols_p, Q, vols_q, type_vols_q, t, loops, mi_o, mi_w, fi, dim)
sim.run()
