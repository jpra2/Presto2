from test34 import MsClassic_mono
import time

class gravidade(MsClassic_mono):
    def __init__(self, ind = False):
        super().__init__(ind = ind)
        self.run_grav()

    def create_flux_vector_pf_gr(self):
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume, adicinando o efeito da gravidade
        """
        t0 = time.time()

        verif_local = 1
        lim4 = 1e-4
        soma = 0
        soma2 = 0
        store_flux_pf = {}

        for volume in self.all_fine_vols:
            #1
            flux = {}
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            centroid_volume = self.mesh_topo_util.get_average_position([volume])
            adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            pvol = self.mb.tag_get_data(self.pf_tag, volume, flat=True)[0]
            for adj in adjs_vol:
                #2
                gid_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                padj = self.mb.tag_get_data(self.pf_tag, adj, flat=True)[0]
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                centroid_adj = self.mesh_topo_util.get_average_position([adj])
                direction = centroid_adj - centroid_volume
                unit = direction/np.linalg.norm(direction)
                #unit = vetor unitario na direcao de direction
                uni = self.unitary(direction)
                # uni = valor positivo do vetor unitario
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/(self.mi)
                grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                q = (grad_p)*keq
                flux[tuple(unit)] = q
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            store_flux_pf[volume] = flux
            flt = sum(flux.values())
            self.mb.tag_set_data(self.flux_fine_pf_tag, volume, flt)
            soma += flt
            if abs(flt) > lim4 and volume not in self.wells:
                verif_local = 0
                print('nao esta dando conservativo na malha fina')
                print(gid_vol)
                print(flt)
                import pdb; pdb.set_trace()
        soma_prod = []
        soma_inj = []
        with open('fluxo_malha_fina.txt', 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
                values = store_flux_pf[volume].values()
                arq.write('gid:{0} , fluxo:{1}\n'.format(gid, sum(values)))

                # print('gid:{0}'.format(gid))
                # print('valor:{0}'.format(sum(values)))
                if volume in self.wells_inj:
                    soma_inj.append(sum(values))
                else:
                    soma_prod.append(sum(values))
                # print('\n')
                soma2 += sum(values)
            arq.write('\n')
            arq.write('soma_inj:{0}\n'.format(sum(soma_inj)))
            arq.write('soma_prod:{0}'.format(sum(soma_prod)))

        print('soma_inj:{0}'.format(sum(soma_inj)))
        print('soma_prod:{0}'.format(sum(soma_prod)))

        print('soma2 : {0}'.format(soma2))
        if abs(soma2) > lim4:
            print('nao esta dando conservativo globalmente')
            import pdb; pdb.set_trace()

        # print('saiu de def create_flux_vector_pf')
        print('\n')

        tf = time.time()
        # import pdb; pdb.set_trace()
        return store_flux_pf

    def mount_lines_5_gr(self, volume, map_id):
        """
        monta as linhas da matriz
        retorna o valor temp_k e o mapeamento temp_id
        map_id = mapeamento local dos elementos
        adiciona o efeito da gravidade
        temp_ids = [] # vetor com ids dados pelo mapeamento
        temp_k = [] # vetor com a permeabilidade equivalente
        temp_kgr = [] # vetor com a permeabilidade equivalente multipicada pelo gama
        temp_hs = [] # vetor com a diferença de altura dos elementos

        """
        #0

        soma2 = 0.0
        soma3 = 0.0
        temp_ids = []
        temp_k = []
        temp_kgr = []
        temp_hs = []
        gid1 = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        volume_centroid = self.mesh_topo_util.get_average_position([volume])
        adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        for adj in adj_volumes:
            #2
            gid2 = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
            temp_ps.append(padj)
            adj_centroid = self.mesh_topo_util.get_average_position([adj])
            direction = adj_centroid - volume_centroid
            altura = adj_centroid[2]
            uni = self.unitary(direction)
            z = uni[2]
            kvol = np.dot(np.dot(kvol,uni),uni)
            kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
            kadj = np.dot(np.dot(kadj,uni),uni)
            keq = self.kequiv(kvol, kadj)
            keq = keq*(np.dot(self.A, uni))/(self.mi*abs(np.dot(direction, uni)))
            if z == 1.0:
                #3
                keq2 = keq*self.gama
                temp_kgr.append(-keq2)
                soma2 = soma2 - keq2
                soma3 = soma3 + (keq2*(self.tz-altura))
                temp_hs.append(self.tz-altura)
            else:
                temp_kgr.append(0.0)
                temp_hs.append(0.0)
            #2
            temp_ids.append(map_id[adj])
            temp_k.append(-keq)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        #1
        soma2 = soma2*(self.tz-volume_centroid[2])
        soma2 = -(soma2 + soma3)
        temp_hs.append(self.tz-volume_centroid[2])
        temp_kgr.append(-sum(temp_kgr))
        temp_k.append(-sum(temp_k))
        temp_ids.append(map_id[volume])
        temp_ps.append(pvol)
        import pdb; pdb.set_trace()

        return temp_k, temp_ids, temp_hs, temp_kgr

    def run_grav(self):
        """
        roda o programa inteiro adicionando o efeito da gravidade
        """

        self.set_contorno()
        self.set_global_problem_gr_vf_3()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, len(self.all_fine_vols_ic))
        self.organize_Pf()
        del self.Pf
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf_all))
        del self.Pf_all

        self.mb.write_file('new_out_mono.vtk')

sim_grav_mono = gravidade(ind = True)
