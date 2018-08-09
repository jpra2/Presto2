import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import time
import sys
import shutil
import os
import random



class Msclassic_bif:

    def __init__(self):
        """
        self.comm = Epetra.PyComm()
        self.mb = core.Core()
        self.mb.load_file('out.h5m')
        self.root_set = self.mb.get_root_set()
        self.mesh_topo_util = topo_util.MeshTopoUtil(self.mb)
        self.all_fine_vols = self.mb.get_entities_by_dimension(self.root_set, 3)
        self.nf = len(self.all_fine_vols)
        self.create_tags(self.mb)
        self.read_structured()
        self.primals = self.mb.get_entities_by_type_and_tag(
                self.root_set, types.MBENTITYSET, np.array([self.primal_id_tag]),
                np.array([None]))
        self.nc = len(self.primals)
        self.ident_primal = []
        for primal in self.primals:
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            self.ident_primal.append(primal_id)
        self.ident_primal = dict(zip(self.ident_primal, range(len(self.ident_primal))))
        self.sets = self.mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, self.collocation_point_tag, np.array([None]))
        self.set_of_collocation_points_elems = set()
        for collocation_point_set in self.sets:
            collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]
            self.set_of_collocation_points_elems.add(collocation_point)
        #self.ident_primal = remapeamento dos ids globais
        self.loops = 1000 # loops totais
        self.t = 1e7 # tempo total de simulacao
        self.mi_w = 1.0 # viscosidade da agua
        self.mi_o = 1.25 # viscosidade do oleo
        self.ro_w = 1.0 # densidade da agua
        self.ro_o = 0.98 # densidade do oleo
        self.gama_w = 1.0 #  peso especifico da agua
        self.gama_o = 0.98 # peso especifico do oleo
        self.gama_ = self.gama_w + self.gama_o
        self.Swi = 0.2 # saturacao inicial para escoamento da agua
        self.Swc = 0.2 # saturacao de agua conata
        self.Sor = 0.2 # saturacao residual de oleo
        self.nw = 2 # expoente da agua para calculo da permeabilidade relativa
        self.no = 2 # expoente do oleo para calculo da permeabilidade relativa
        """
        self.read_perms_and_phi_spe10()
        self.set_k() # seta a permeabilidade em cada volume
        self.set_fi() # seta a porosidade em cada volume
        self.get_wells() # obtem os gids dos volumes que sao pocos
        self.read_perm_rel() # le o arquivo txt perm_rel.txt
        gids = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols , flat = True)
        self.map_gids_in_all_fine_vols = dict(zip(gids, self.all_fine_vols)) # mapeamento dos gids nos elementos

        self.neigh_wells_d = [] #volumes da malha fina vizinhos aos pocos de pressao prescrita
        for volume in self.wells:

            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if volume in self.wells_d:

                adjs_volume = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                for adj in adjs_volume:

                    global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    if (adj not in self.wells_d) and (adj not in self.neigh_wells_d):

                        self.neigh_wells_d.append(adj)

        self.all_fine_vols_ic = set(self.all_fine_vols) - set(self.wells_d)
        # self.all_volumes_ic =  volumes da malha fina que sao incognitas
        self.map_vols_ic = dict(zip(list(self.all_fine_vols_ic), range(len(self.all_fine_vols_ic)))) # mapeamento dos elementos que sao incognitas
        self.map_vols_ic_2 = dict(zip(range(len(self.all_fine_vols_ic)), list(self.all_fine_vols_ic))) # mapeamento contrario
        self.nf_ic = len(self.all_fine_vols_ic) # numero de icognitas
        self.principal = '/elliptic'
        self.caminho1 = '/elliptic/simulacoes/bifasico'
        self.caminho2 = '/elliptic/simulacoes'
        self.caminho3 = '/elliptic/backup_simulacoes'
        self.caminho4 = '/elliptic/backup_simulacoes/bifasico'
        self.caminho5 = '/elliptic/backup_simulacoes/bifasico/pasta0'
        arq1 = 'back.txt'

        # ##### abaixo esta o comando para deletar a pasta backup_simulacoes ##########
        # shutil.rmtree(self.caminho3)
        # sys.exit(0)
        # import pdb; pdb.set_trace()
        # ############################################################################

        if os.path.exists(self.caminho2):
            if os.path.exists(self.caminho1):
                shutil.rmtree(self.caminho1)
                os.makedirs(self.caminho1)
            else:
                os.makedirs(self.caminho1)
        else:
            os.makedirs(self.caminho1)

        if os.path.exists(self.caminho3):
            if os.path.exists(self.caminho4):
                os.chdir(self.caminho4)
                if arq1 in os.listdir():
                    with open(arq1, 'r') as arq:
                        text = arq.readline()
                        num_sim = int(text) + 1
                    with open(arq1, 'w') as arq:
                        arq.write('{0}'.format(num_sim))

                    self.pasta = '/elliptic/backup_simulacoes/bifasico/pasta{0}'.format(num_sim)
                    # os.makedirs(self.pasta)

                else:
                    with open(arq1, 'w') as arq:
                        arq.write('{0}'.format(int(0)))
                    self.pasta = self.caminho5
            else:
                os.makedirs(self.caminho4)
                os.chdir(self.caminho4)
                with open(arq1, 'w') as arq:
                    arq.write('{0}'.format(int(0)))
                self.pasta = self.caminho5
        else:
            os.makedirs(self.caminho4)
            os.chdir(self.caminho4)
            with open(arq1, 'w') as arq:
                arq.write('{0}'.format(int(0)))
            self.pasta = self.caminho5

        os.chdir(self.caminho1)


    def calculate_local_problem_het(self, elems, lesser_dim_meshsets, support_vals_tag):
        std_map = Epetra.Map(len(elems), 0, self.comm)
        linear_vals = np.arange(0, len(elems))
        id_map = dict(zip(elems, linear_vals))
        boundary_elms = set()

        b = Epetra.Vector(std_map)
        x = Epetra.Vector(std_map)

        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        for ms in lesser_dim_meshsets:
            lesser_dim_elems = self.mb.get_entities_by_handle(ms)
            for elem in lesser_dim_elems:
                if elem in boundary_elms:
                    continue
                boundary_elms.add(elem)
                idx = id_map[elem]
                A.InsertGlobalValues(idx, [1], [idx])
                b[idx] = self.mb.tag_get_data(support_vals_tag, elem, flat=True)[0]

        for elem in (set(elems) ^ boundary_elms):
            k_elem = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
            lamb_w_elem = self.mb.tag_get_data(self.lamb_w_tag, elem)[0][0]
            lamb_o_elem = self.mb.tag_get_data(self.lamb_o_tag, elem)[0][0]
            centroid_elem = self.mesh_topo_util.get_average_position([elem])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(
                np.asarray([elem]), 2, 3, 0)
            values = []
            ids = []
            for adj in adj_volumes:
                if adj in id_map:
                    k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_elem
                    uni = self.unitary(direction)
                    k_elem = np.dot(np.dot(k_elem,uni),uni)
                    k_elem = k_elem*(lamb_w_elem + lamb_o_elem)
                    k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    k_adj = np.dot(np.dot(k_adj,uni),uni)
                    lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                    lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                    k_adj = k_adj*(lamb_w_adj + lamb_o_adj)
                    keq = self.kequiv(k_elem, k_adj)
                    #keq = keq/(np.dot(self.h2, uni))
                    keq = keq*(np.dot(self.A, uni)/(np.dot(self.h, uni)))
                    values.append(keq)
                    ids.append(id_map[adj])
                    k_elem = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
            values.append(-sum(values))
            idx = id_map[elem]
            ids.append(idx)
            A.InsertGlobalValues(idx, values, ids)

        A.FillComplete()

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        # AZ_last, AZ_summary, AZ_warnings
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-9)

        self.mb.tag_set_data(support_vals_tag, elems, np.asarray(x))

    def calculate_p_end(self):

        for volume in self.wells:
            global_volume = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if global_volume in self.wells_d:
                index = self.wells_d.index(global_volume)
                pms = self.set_p[index]
                mb.tag_set_data(self.pms_tag, volume, pms)

    def calculate_prolongation_op_het(self):

        zeros = np.zeros(self.nf)
        std_map = Epetra.Map(self.nf, 0, self.comm)
        self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        sets = self.mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, self.collocation_point_tag, np.array([None]))

        i = 0

        my_pairs = set()

        for collocation_point_set in sets:

            i += 1
            childs = self.mb.get_child_meshsets(collocation_point_set)
            collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]
            primal_elem = self.mb.tag_get_data(self.fine_to_primal_tag, collocation_point,
                                           flat=True)[0]
            primal_id = self.mb.tag_get_data(self.primal_id_tag, int(primal_elem), flat=True)[0]

            primal_id = self.ident_primal[primal_id]

            support_vals_tag = self.mb.tag_get_handle(
                "TMP_SUPPORT_VALS {0}".format(primal_id), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)

            self.mb.tag_set_data(support_vals_tag, self.all_fine_vols, zeros)
            self.mb.tag_set_data(support_vals_tag, collocation_point, 1.0)

            for vol in childs:
                elems_vol = self.mb.get_entities_by_handle(vol)
                c_faces = self.mb.get_child_meshsets(vol)

                for face in c_faces:
                    elems_fac = self.mb.get_entities_by_handle(face)
                    c_edges = self.mb.get_child_meshsets(face)

                    for edge in c_edges:
                        elems_edg = self.mb.get_entities_by_handle(edge)
                        c_vertices = self.mb.get_child_meshsets(edge)
                        # a partir desse ponto op de prolongamento eh preenchido
                        self.calculate_local_problem_het(
                            elems_edg, c_vertices, support_vals_tag)

                    self.calculate_local_problem_het(
                        elems_fac, c_edges, support_vals_tag)

                self.calculate_local_problem_het(
                    elems_vol, c_faces, support_vals_tag)


                vals = self.mb.tag_get_data(support_vals_tag, elems_vol, flat=True)
                gids = self.mb.tag_get_data(self.global_id_tag, elems_vol, flat=True)
                primal_elems = self.mb.tag_get_data(self.fine_to_primal_tag, elems_vol,
                                               flat=True)

                for val, gid in zip(vals, gids):
                    if (gid, primal_id) not in my_pairs:
                        if val == 0.0:
                            pass
                        else:
                            self.trilOP.InsertGlobalValues([gid], [primal_id], val)

                        my_pairs.add((gid, primal_id))

    def calculate_restriction_op(self):

        std_map = Epetra.Map(self.nf, 0, self.comm)
        self.trilOR = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        for primal in self.primals:

            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id]
            restriction_tag = self.mb.tag_get_handle(
                            "RESTRICTION_PRIMAL {0}".format(primal_id), 1, types.MB_TYPE_INTEGER,
                            True, types.MB_TAG_SPARSE)

            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)

            self.mb.tag_set_data(
                self.elem_primal_id_tag,
                fine_elems_in_primal,
                np.repeat(primal_id, len(fine_elems_in_primal)))

            gids = self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True)
            self.trilOR.InsertGlobalValues(primal_id, np.repeat(1, len(gids)), gids)

            self.mb.tag_set_data(restriction_tag, fine_elems_in_primal, np.repeat(1, len(fine_elems_in_primal)))

        self.trilOR.FillComplete()

        """for i in range(len(primals)):
            p = trilOR.ExtractGlobalRowCopy(i)
            print(p[0])
            print(p[1])
            print('\n')"""

    def calculate_restriction_op_2(self):
        """
        operador de restricao excluindo as colunas dos volumes com pressao prescrita
        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic), 0, self.comm)
        self.trilOR = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols_ic, flat=True)
        for primal in self.primals:
            #1
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id]
            restriction_tag = self.mb.tag_get_handle(
                            "RESTRICTION_PRIMAL {0}".format(primal_id), 1, types.MB_TYPE_INTEGER,
                            True, types.MB_TAG_SPARSE)
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            self.mb.tag_set_data(
                self.elem_primal_id_tag,
                fine_elems_in_primal,
                np.repeat(primal_id, len(fine_elems_in_primal)))
            elems_ic = self.all_fine_vols_ic & set(fine_elems_in_primal)
            local_map = []
            for elem in elems_ic:
                #2
                local_map.append(self.map_vols_ic[elem])
            #1
            self.trilOR.InsertGlobalValues(primal_id, np.repeat(1, len(local_map)), local_map)
            #gids = self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True)
            #self.trilOR.InsertGlobalValues(primal_id, np.repeat(1, len(gids)), gids)
            self.mb.tag_set_data(restriction_tag, fine_elems_in_primal, np.repeat(1, len(fine_elems_in_primal)))
        #0
        self.trilOR.FillComplete()
        """for i in range(len(self.primals)):
            p = self.trilOR.ExtractGlobalRowCopy(i)
            print(p[0])
            print(p[1])
            print('\n')"""

    def calculate_sat(self):
        """
        calcula a saturacao do passo de tempo corrente
        """
        t1 = time.time()
        lim = 10**(-6)

        for volume in self.all_fine_vols:
            gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if gid in self.wells_d:
                tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)[0][0]
                if tipo_de_poco == 1:
                    continue
                else:
                    pass
            div = self.div_upwind_3(volume, self.pf_tag)
            fi = 0.3 #self.mb.tag_get_data(self.fi_tag, volume)[0][0]
            sat1 = self.mb.tag_get_data(self.sat_tag, volume)[0][0]
            sat = sat1 + div*(self.delta_t/(fi*self.V))
            if sat > 1.0:
                print('saturacao maior que 1 na funcao calculate_sat')
                import pdb; pdb.set_trace()
            #if abs(div) < lim or sat1 == (1 - self.Sor) or sat < sat1:
            #if abs(div) < lim or sat1 == (1 - self.Sor):
            if abs(div) < lim or sat1 == 0.8:
                continue

            #elif sat > (1 - self.Sor):
            elif sat > 0.8:
                #sat = 1 - self.Sor
                print("Sat > 0.8")
                print(sat)
                print('gid')
                print(gid)
                print('\n')
                sat = 0.8

            #elif sat < 0 or sat > (1 - self.Sor):
            elif sat < 0 or sat > 0.8:
                print('Erro: saturacao invalida')
                print('Saturacao: {0}'.format(sat))
                print('Saturacao anterior: {0}'.format(sat1))
                print('div: {0}'.format(div))
                print('gid: {0}'.format(gid))
                print('fi: {0}'.format(fi))
                print('V: {0}'.format(self.V))
                print('delta_t: {0}'.format(self.delta_t))
                print('loop: {0}'.format(self.loop))


                sys.exit(0)

            self.mb.tag_set_data(self.sat_tag, volume, sat)

        t2 = time.time()

    def calculate_sat_2(self):
        """
        calcula a saturacao do passo de tempo corrente
        """
        t1 = time.time()
        lim = 10**(-6)

        for volume in self.all_fine_vols:
            gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if volume in self.wells_inj:
                continue
            qw = self.mb.tag_get_data(self.flux_w_tag, volume, flat=True)[0]
            fi = self.mb.tag_get_data(self.fi_tag, volume)[0][0]
            sat1 = self.mb.tag_get_data(self.sat_tag, volume)[0][0]
            sat = sat1 + qw*(self.delta_t/(fi*self.V))
            if sat1 > sat:
                print('erro na saturacao')
                print('sat1 > sat')
                import pdb; pdb.set_trace()
            # print('gid:{0}'.format(gid))
            # print('sat1:{0}'.format(sat1))
            # print('sat:{0}'.format(sat))
            # print('qw:{0}'.format(qw))
            # print('const:{0}'.format(self.delta_t/(fi*self.V)))
            # print('res:.{0}'.format(qw*(self.delta_t/(fi*self.V))))

            # import pdb; pdb.set_trace()
            # if sat > 0.8:
            #     print('saturacao maior que 0.8 na funcao calculate_sat')
                # import pdb; pdb.set_trace()
            #if abs(div) < lim or sat1 == (1 - self.Sor) or sat < sat1:
            #if abs(div) < lim or sat1 == (1 - self.Sor):
            elif abs(qw) < lim or sat1 == 0.8:
                continue

            #elif sat > (1 - self.Sor):
            elif sat > 0.8:
                #sat = 1 - self.Sor
                print("Sat > 0.8")
                print(sat)
                print('gid')
                print(gid)
                print('\n')
                # import pdb; pdb.set_trace()
                sat = 0.8

            #elif sat < 0 or sat > (1 - self.Sor):
            elif sat < 0 or sat > 0.8:
                print('Erro: saturacao invalida')
                print('Saturacao: {0}'.format(sat))
                print('Saturacao anterior: {0}'.format(sat1))
                print('div: {0}'.format(div))
                print('gid: {0}'.format(gid))
                print('fi: {0}'.format(fi))
                print('V: {0}'.format(self.V))
                print('delta_t: {0}'.format(self.delta_t))
                print('loop: {0}'.format(self.loop))
                import pdb; pdb.set_trace()

                sys.exit(0)

            else:
                pass

            self.mb.tag_set_data(self.sat_tag, volume, sat)

        t2 = time.time()
        print('tempo calculo saturacao loop_{0}: {1}'.format(self.loop, t2-t1))

    def cfl(self):
        """
        cfl usando fluxo maximo
        """

        cfl = 0.1

        self.delta_t = cfl*(self.fimin*self.V)/float(self.qmax*self.dfdsmax)

    def cfl_2(self, vmax, h, dfds):
        """
        cfl usando velocidade maxima
        """
        cfl = 1.0

        self.delta_t = (cfl*h)/float(vmax*dfds)

    def create_flux_vector_pf(self):
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume
        """
        lim = 1e-5
        self.dfdsmax = 0
        self.fimin = 10
        self.qmax = 0
        self.store_velocity_pf = {}
        self.store_flux_pf = {}
        for primal in self.primals:
            #1
            primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id1]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_interface, volumes_in_primal = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id1, flag = 1)
            for volume in fine_elems_in_primal:
                #2
                list_keq = []
                list_p = []
                list_gid = []
                list_keq3 = []
                list_gidsadj = []
                qw3 = []
                qw = 0
                flux = {}
                velocity = {}
                fi = self.mb.tag_get_data(self.fi_tag, volume, flat=True)[0]
                if fi < self.fimin:
                    self.fimin = fi
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume, flat=True)[0]
                lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume, flat=True)[0]
                fw_vol = self.mb.tag_get_data(self.fw_tag, volume, flat=True)[0]
                sat_vol = self.mb.tag_get_data(self.sat_tag, volume, flat=True)[0]
                centroid_volume = self.mesh_topo_util.get_average_position([volume])
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                pvol = self.mb.tag_get_data(self.pf_tag, volume, flat=True)[0]
                for adj in adjs_vol:
                    #3
                    gid_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    sat_adj = self.mb.tag_get_data(self.sat_tag, adj, flat=True)[0]
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
                    lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj, flat=True)[0]
                    lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj, flat=True)[0]
                    fw_adj = self.mb.tag_get_data(self.fw_tag, adj, flat=True)[0]
                    kvol = kvol*(lamb_w_vol + lamb_o_vol)
                    kadj = kadj*(lamb_w_adj + lamb_o_adj)
                    keq = self.kequiv(kvol, kadj)

                    list_keq.append(keq)
                    list_p.append(padj)
                    list_gid.append(gid_adj)

                    keq2 = keq
                    keq3 = (kvol + kadj)/2.0
                    keq = keq*(np.dot(self.A, uni))
                    #pvol2 = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    #padj2 = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                    grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                    #grad_p2 = (padj2 - pvol2)/float(abs(np.dot(direction, uni)))
                    q = (grad_p)*keq
                    if grad_p < 0:
                        #4
                        fw = fw_vol
                        qw += (fw*grad_p*kvol*(np.dot(self.A, uni)))
                        qw3.append(fw*grad_p*keq3*(np.dot(self.A, uni)))
                    else:
                        fw = fw_adj
                        qw += (fw*grad_p*kadj*(np.dot(self.A, uni)))
                        qw3.append(fw*grad_p*keq3*(np.dot(self.A, uni)))

                    if gid_adj > gid_vol:
                        v = -(grad_p)*keq2
                    else:
                        v = (grad_p)*keq2

                    flux[tuple(unit)] = q
                    velocity[tuple(unit)] = v
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    if abs(sat_adj - sat_vol) < lim or abs(fw_adj -fw_vol) < lim:
                        continue
                    dfds = abs((fw_adj - fw_vol)/(sat_adj - sat_vol))
                    # print('aqui')
                    # print(gid_vol)
                    # print(gid_adj)
                    # print(fw_adj - fw_vol)
                    # print(sat_adj - sat_vol)
                    # print(dfds)
                    if dfds > self.dfdsmax:
                        self.dfdsmax = dfds

                #2
                list_keq.append(-sum(list_keq))
                list_p.append(pvol)
                list_gid.append(gid_vol)

                list_keq = np.array(list_keq)
                list_p = np.array(list_p)
                resultado = sum(list_keq*list_p)

                # print(gid_vol)
                # print(velocity)
                # print('\n')
                # import pdb; pdb.set_trace()
                self.store_velocity_pf[volume] = velocity
                self.store_flux_pf[volume] = flux
                flt = sum(flux.values())
                self.mb.tag_set_data(self.flux_fine_pf_tag, volume, flt)

                if abs(sum(flux.values())) > lim and volume not in self.wells:
                    print('nao esta dando conservativo na malha fina')
                    print(gid_vol)
                    print(sum(flux.values()))

                qmax = max(list(map(abs, flux.values())))
                if qmax > self.qmax:
                    self.qmax = qmax
                if volume in self.wells_prod:
                    qw_out = sum(flux.values())*fw_vol
                    qo_out = sum(flux.values())*(1 - fw_vol)
                    self.prod_o.append(qo_out)
                    self.prod_w.append(qw_out)
                    qw = qw - qw_out
                if qw < 0 and volume not in self.wells_inj:
                    print(gid_vol)
                    print('qw < 0')
                    import pdb; pdb.set_trace()

                self.mb.tag_set_data(self.flux_w_tag, volume, qw)

                # print(self.dfdsmax)
                # print(sum(flux.values()))
                # print(sum(qw))
                # print(sum(qw3))
                # print('\n')

        soma_inj = []
        soma_prod = []
        soma2 = 0
        with open('fluxo_malha_fina_bif{0}.txt'.format(self.loop), 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
                values = self.store_flux_pf[volume].values()
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
            arq.write('soma_prod:{0}\n'.format(sum(soma_prod)))
            arq.write('tempo:{0}'.format(self.tempo))


    def create_flux_vector_pms(self):
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume
        """
        lim = 1e-5
        self.dfdsmax = 0
        self.fimin = 10
        self.qmax = 0
        self.store_velocity = {}
        self.store_flux = {}
        for primal in self.primals:
            #1
            primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id1]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_interface, volumes_in_primal = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id1, flag = 1)
            for volume in fine_elems_in_primal:
                #2
                list_keq3 = []
                list_gidsadj = []
                qw3 = []
                qw = 0
                flux = {}
                velocity = {}
                fi = self.mb.tag_get_data(self.fi_tag, volume, flat=True)[0]
                if fi < self.fimin:
                    self.fimin = fi
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume, flat=True)[0]
                lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume, flat=True)[0]
                fw_vol = self.mb.tag_get_data(self.fw_tag, volume, flat=True)[0]
                sat_vol = self.mb.tag_get_data(self.sat_tag, volume, flat=True)[0]
                centroid_volume = self.mesh_topo_util.get_average_position([volume])
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                for adj in adjs_vol:
                    #3
                    gid_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    sat_adj = self.mb.tag_get_data(self.sat_tag, adj, flat=True)[0]
                    if adj in volumes_in_interface:
                        #4
                        pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                        padj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                    #3
                    else:
                        #4
                        pvol = self.mb.tag_get_data(self.pcorr_tag, volume, flat=True)[0]
                        padj = self.mb.tag_get_data(self.pcorr_tag, adj, flat=True)[0]
                    #3
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_volume
                    unit = direction/np.linalg.norm(direction)
                    #unit = vetor unitario na direcao de direction
                    uni = self.unitary(direction)
                    # uni = valor positivo do vetor unitario
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj, flat=True)[0]
                    lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj, flat=True)[0]
                    fw_adj = self.mb.tag_get_data(self.fw_tag, adj, flat=True)[0]
                    kvol = kvol*(lamb_w_vol + lamb_o_vol)
                    kadj = kadj*(lamb_w_adj + lamb_o_adj)
                    keq = self.kequiv(kvol, kadj)
                    keq2 = keq
                    keq3 = (kvol + kadj)/2.0
                    keq = keq*(np.dot(self.A, uni))
                    pvol2 = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    padj2 = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                    grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                    grad_p2 = (padj2 - pvol2)/float(abs(np.dot(direction, uni)))
                    q = (grad_p)*keq
                    if grad_p < 0:
                        #4
                        fw = fw_vol
                        qw += (fw*grad_p*kvol*(np.dot(self.A, uni)))
                        qw3.append(fw*grad_p*keq3*(np.dot(self.A, uni)))
                    else:
                        fw = fw_adj
                        qw += (fw*grad_p*kadj*(np.dot(self.A, uni)))
                        qw3.append(fw*grad_p*keq3*(np.dot(self.A, uni)))

                    if gid_adj > gid_vol:
                        v = -(grad_p2)*keq2
                    else:
                        v = (grad_p2)*keq2

                    flux[tuple(unit)] = q
                    velocity[tuple(unit)] = v
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    if abs(sat_adj - sat_vol) < lim or abs(fw_adj -fw_vol) < lim:
                        continue
                    dfds = abs((fw_adj - fw_vol)/(sat_adj - sat_vol))
                    # print('aqui')
                    # print(gid_vol)
                    # print(gid_adj)
                    # print(fw_adj - fw_vol)
                    # print(sat_adj - sat_vol)
                    # print(dfds)
                    if dfds > self.dfdsmax:
                        self.dfdsmax = dfds
                #2
                # print(gid_vol)
                # print(velocity)
                # print('\n')
                # import pdb; pdb.set_trace()
                self.store_velocity[volume] = velocity
                self.store_flux[volume] = flux
                self.mb.tag_set_data(self.flux_fine_pms_tag, volume, sum(flux.values()))

                if abs(sum(flux.values())) > lim and volume not in self.wells:
                    print('nao esta dando conservativo o fluxo multiescala')
                    print(gid_vol)
                    print(sum(flux.values()))
                    import pdb; pdb.set_trace()

                qmax = max(list(map(abs, flux.values())))
                if qmax > self.qmax:
                    self.qmax = qmax
                if volume in self.wells_prod:
                    qw_out = sum(flux.values())*fw_vol
                    qo_out = sum(flux.values())*(1 - fw_vol)
                    self.prod_o.append(qo_out)
                    self.prod_w.append(qw_out)
                    qw = qw - qw_out
                if qw < 0 and volume not in self.wells_inj:
                    print(gid_vol)
                    print('qw < 0')
                    import pdb; pdb.set_trace()
                    sys.exit(0)

                self.mb.tag_set_data(self.flux_w_tag, volume, qw)

                # print(self.dfdsmax)
                # print(sum(flux.values()))
                # print(sum(qw))
                # print(sum(qw3))
                # print('\n')

        soma_inj = []
        soma_prod = []
        soma2 = 0
        with open('fluxo_multiescala_bif{0}.txt'.format(self.loop), 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
                values = self.store_flux[volume].values()
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
            arq.write('soma_prod:{0}\n'.format(sum(soma_prod)))
            arq.write('tempo:{0}'.format(self.tempo))


    def create_tags(self, mb):

        self.flux_coarse_tag = mb.tag_get_handle(
                        "FLUX_COARSE", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.prod_tag = mb.tag_get_handle(
                        "PROD", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)


        self.lbt_tag = mb.tag_get_handle(
                        "LBT", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.fw_tag = mb.tag_get_handle(
                        "FW", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.vel_tag = mb.tag_get_handle(
                        "VEL", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.Pc2_tag = mb.tag_get_handle(
                        "PC2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pf2_tag = mb.tag_get_handle(
                        "PF2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.err_tag = mb.tag_get_handle(
                        "ERRO", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.err2_tag = mb.tag_get_handle(
                        "ERRO_2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pf_tag = mb.tag_get_handle(
                        "PF", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.k_tag = mb.tag_get_handle(
                        "K", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.contorno_tag = mb.tag_get_handle(
                        "CONTORNO", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pc_tag = mb.tag_get_handle(
                        "PC", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pms_tag = mb.tag_get_handle(
                        "PMS", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pms2_tag = mb.tag_get_handle(
                        "PMS2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.flux_w_tag = mb.tag_get_handle(
                        "FLUX_W", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.flux_fine_pms_tag = mb.tag_get_handle(
                        "FLUX_FINE_PMS", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.flux_fine_pf_tag = mb.tag_get_handle(
                        "FLUX_FINE_PF", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.p_tag = mb.tag_get_handle(
                        "P", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pcorr_tag = mb.tag_get_handle(
                        "P_CORR", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.perm_tag = mb.tag_get_handle(
                        "PERM", 9, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.global_id_tag = mb.tag_get_handle("GLOBAL_ID")

        self.collocation_point_tag = mb.tag_get_handle("COLLOCATION_POINT")

        self.elem_primal_id_tag = mb.tag_get_handle(
            "FINE_PRIMAL_ID", 1, types.MB_TYPE_INTEGER, True,
            types.MB_TAG_SPARSE)

        self.sat_tag = mb.tag_get_handle(
                        "SAT", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.fi_tag = mb.tag_get_handle(
                        "FI", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.lamb_w_tag = mb.tag_get_handle(
                        "LAMB_W", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.lamb_o_tag =  mb.tag_get_handle(
                        "LAMB_O", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.primal_id_tag = mb.tag_get_handle("PRIMAL_ID")
        self.fine_to_primal_tag = mb.tag_get_handle("FINE_TO_PRIMAL")
        self.valor_da_prescricao_tag = mb.tag_get_handle("VALOR_DA_PRESCRICAO")
        self.tipo_de_prescricao_tag = mb.tag_get_handle("TIPO_DE_PRESCRICAO")
        self.wells_tag = mb.tag_get_handle("WELLS")
        self.tipo_de_poco_tag = mb.tag_get_handle("TIPO_DE_POCO")

    def Dirichlet_problem(self):
        """
        recalculo das pressoes dentro dos primais usando como condicao de contorno
        pressao prescrita nos volumes da interface de cada primal
        """
        #0
        colocation_points = self.mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, self.collocation_point_tag, np.array([None]))
        sets = []
        for col in colocation_points:
            #1
            #col = mb.get_entities_by_handle(col)[0]
            sets.append(self.mb.get_entities_by_handle(col)[0])
        #0
        sets = set(sets)
        for primal in self.primals:
            #1
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_primal = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id, flag = 2)
            all_volumes = list(fine_elems_in_primal)
            all_volumes_ic = self.all_fine_vols_ic & set(all_volumes)
            gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, all_volumes_ic, flat=True)
            # gids_vols_ic = volumes no primal que sao icognitas
            # ou seja volumes no primal excluindo os que tem pressao prescrita
            map_volumes = dict(zip(gids_vols_ic, range(len(gids_vols_ic))))
            # map_volumes = mapeamento local
            std_map = Epetra.Map(len(all_volumes_ic), 0, self.comm)
            b = Epetra.Vector(std_map)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            dim = len(all_volumes_ic)
            # b_np = np.zeros(dim)
            # A_np = np.zeros((dim, dim))
            for volume in all_volumes_ic:
                #2
                soma = 0
                temp_id = []
                temp_k = []
                volume_centroid = self.mesh_topo_util.get_average_position([volume])
                adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
                lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
                if (volume in sets) or (volume in volumes_in_primal):
                    #3
                    temp_k.append(1.0)
                    temp_id.append(map_volumes[global_volume])
                    b[map_volumes[global_volume]] = self.mb.tag_get_data(self.pms_tag, volume)[0]
                    # b_np[map_volumes[global_volume]] = self.mb.tag_get_data(self.pms_tag, volume)[0]
                #2
                else:
                    #3
                    for adj in adj_volumes:
                        #4
                        global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                        adj_centroid = self.mesh_topo_util.get_average_position([adj])
                        direction = adj_centroid - volume_centroid
                        uni = self.unitary(direction)
                        kvol = np.dot(np.dot(kvol,uni),uni)
                        kvol = kvol*(lamb_w_vol + lamb_o_vol)
                        kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        kadj = np.dot(np.dot(kadj,uni),uni)
                        lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                        lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                        kadj = kadj*(lamb_w_adj + lamb_o_adj)
                        keq = self.kequiv(kvol, kadj)
                        keq = keq*(np.dot(self.A, uni))/(np.dot(self.h, uni))
                        soma = soma + keq
                        if global_adj in self.wells_d:
                            #5
                            index = self.wells_d.index(global_adj)
                            b[map_volumes[global_volume]] += self.set_p[index]*(keq)
                            # b_np[map_volumes[global_volume]] += self.set_p[index]*(keq)
                        #4
                        else:
                            #5
                            temp_id.append(map_volumes[global_adj])
                            temp_k.append(-keq)
                        #4
                        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    #3
                    temp_k.append(soma)
                    temp_id.append(map_volumes[global_volume])
                    if global_volume in self.wells_n:
                        #4
                        index = self.wells_n.index(global_volume)
                        tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)[0]
                        if tipo_de_poco == 1:
                            #5
                            b[map_volumes[global_volume]] += self.set_q[index]
                            # b_np[map_volumes[global_volume]] += self.set_q[index]
                        #4
                        else:
                            #5
                            b[map_volumes[global_volume]] += -self.set_q[index]
                            # b_np[map_volumes[global_volume]] += -self.set_q[index]
                #2
                A.InsertGlobalValues(map_volumes[global_volume], temp_k, temp_id)
                # A_np[map_volumes[global_volume], temp_id] = temp_k
            #1
            A.FillComplete()
            x = self.solve_linear_problem(A, b, dim)
            # x_np = np.linalg.solve(A_np, b_np)
            for volume in all_volumes_ic:
                #2
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                self.mb.tag_set_data(self.pcorr_tag, volume, x[map_volumes[global_volume]])
                # self.mb.tag_set_data(self.pms2_tag, volume, x_np[map_volumes[global_volume]])
            #1
            for volume in set(all_volumes) - all_volumes_ic:
                #2
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                index = self.wells_d.index(global_volume)
                p = self.set_p[index]
                self.mb.tag_set_data(self.pcorr_tag, volume, p)
                # self.mb.tag_set_data(self.pms2_tag, volume, p)

    def div_max(self, p_tag):
        q2 = 0.0
        fi = 0.0
        for volume in self.all_fine_vols:
            soma1 = 0.0
            soma2 = 0.0
            pvol = mb.tag_get_data(p_tag, volume)[0][0]
            adjs_vol = mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            volume_centroid = mesh_topo_util.get_average_position([volume])
            global_volume = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            lamb_w_vol = mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = mb.tag_get_data(self.lamb_o_tag, volume)[0][0]

            for adj in adjs_vol:
                padj = mb.tag_get_data(p_tag, adj)[0][0]
                adj_centroid = mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                lamb_w_adj = mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                lamb_o_adj = mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kvol = kvol*(lamb_w_vol + lamb_o_vol)
                kadj = mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                kadj = kadj*(lamb_w_adj + lamb_o_adj)
                keq = self.kequiv(kvol, kadj)
                keq = keq/(np.dot(self.h2, uni))
                soma1 = soma1 - keq
                soma2 = soma2 + keq*padj
                kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

            soma1 = soma1*pvol
            q = soma1 + soma2
            if abs(q) > abs(q2):
                q2 = q
                fi = mb.tag_get_data(self.fi_tag, volume)[0][0]

        return abs(q2), fi

    def div_max_2(self, p_tag):
        q2 = 0.0
        fi = 0.0
        for volume in self.all_fine_vols:
            q = 0.0
            pvol = mb.tag_get_data(p_tag, volume)[0][0]
            adjs_vol = mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            volume_centroid = mesh_topo_util.get_average_position([volume])
            global_volume = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            lamb_w_vol = mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = mb.tag_get_data(self.lamb_o_tag, volume)[0][0]

            for adj in adjs_vol:
                padj = mb.tag_get_data(p_tag, adj)[0][0]
                adj_centroid = mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                lamb_w_adj = mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                lamb_o_adj = mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kvol = kvol*(lamb_w_vol + lamb_o_vol)
                kadj = mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                kadj = kadj*(lamb_w_adj + lamb_o_adj)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/(np.dot(self.h, uni))
                q = q + keq*(padj - pvol)
                kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

            if abs(q) > q2:
                q2 = abs(q)
                fi = mb.tag_get_data(self.fi_tag, volume)[0][0]

        return q2, fi

    def div_max_3(self, p_tag):
        """
        Verifica qual é o fluxo maximo de agua que sai do volume de controle multiplicado pelo dfds
        dfds = variacao do fluxo fracionario com a saturacao
        """
        lim = 10**(-12)
        q2 = 0.0
        fi = 0.0
        fi2 = 0.0
        dfds2 = 0
        for volume in self.all_fine_vols:
            q = 0.0
            pvol = self.mb.tag_get_data(p_tag, volume)[0][0]
            adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
            sat_vol = self.mb.tag_get_data(self.sat_tag, volume)[0][0]
            fi = self.mb.tag_get_data(self.fi_tag, volume)[0][0]
            if fi > fi2:
                fi2 = fi

            for adj in adjs_vol:
                padj = self.mb.tag_get_data(p_tag, adj)[0][0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kvol = kvol*(lamb_w_vol + lamb_o_vol)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                kadj = kadj*(lamb_w_adj + lamb_o_adj)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/(np.dot(self.h, uni))
                sat_adj = self.mb.tag_get_data(self.sat_tag, adj)[0][0]
                if abs(sat_adj - sat_vol) < lim:
                    continue
                dfds = ((lamb_w_adj/(lamb_w_adj+lamb_o_adj)) - (lamb_w_vol/(lamb_w_vol+lamb_o_vol)))/float((sat_adj - sat_vol))
                q = abs(dfds*keq*(padj - pvol))
                if q > q2:
                    q2 = q
                    dfds2 = abs(dfds)
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

        return q2, fi2

    def div_upwind_1(self, volume, p_tag):

        """
        a mobilidade da interface é dada pelo volume com a pressao maior dif fin

        """

        soma1 = 0.0
        soma2 = 0.0
        pvol = mb.tag_get_data(p_tag, volume)[0][0]
        adjs_vol = mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        volume_centroid = mesh_topo_util.get_average_position([volume])
        global_volume = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        lamb_w_vol = mb.tag_get_data(self.lamb_w_tag, volume)[0][0]

        for adj in adjs_vol:
            padj = mb.tag_get_data(p_tag, adj)[0][0]
            adj_centroid = mesh_topo_util.get_average_position([adj])
            direction = adj_centroid - volume_centroid
            uni = self.unitary(direction)
            kvol = np.dot(np.dot(kvol,uni),uni)
            kadj = mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
            kadj = np.dot(np.dot(kadj,uni),uni)
            lamb_w_adj = mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
            grad_p = padj - pvol

            if grad_p > 0:
                keq = (lamb_w_adj*kadj)/(np.dot(self.h2, uni))
            else:
                keq = (lamb_w_vol*kvol)/(np.dot(self.h2, uni))

            soma1 = soma1 + keq
            soma2 = soma2 + keq*padj
            kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        soma1 = -soma1*pvol
        q = soma1 + soma2

        return q

    def div_upwind_2(self, volume, p_tag):

        """
        calcula o fluxo total que entra no volume para calcular a saturacao
        a mobilidade da interface é dada pelo volume com a pressao maior
        """

        q = 0.0

        pvol = mb.tag_get_data(p_tag, volume)[0][0]
        adjs_vol = mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        volume_centroid = mesh_topo_util.get_average_position([volume])
        global_volume = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        lamb_w_vol = mb.tag_get_data(self.lamb_w_tag, volume)[0][0]

        for adj in adjs_vol:
            padj = mb.tag_get_data(p_tag, adj)[0][0]
            adj_centroid = mesh_topo_util.get_average_position([adj])
            direction = adj_centroid - volume_centroid
            uni = self.unitary(direction)
            kvol = np.dot(np.dot(kvol,uni),uni)
            kadj = mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
            kadj = np.dot(np.dot(kadj,uni),uni)
            lamb_w_adj = mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
            grad_p = (padj - pvol)/float((np.dot(self.h, uni)))

            if grad_p > 0:
                # keq = (lamb_w_adj*kadj*(np.dot(self.A, uni)))/(np.dot(self.h, uni))
                keq = lamb_w_adj*kadj

            else:
                # keq = (lamb_w_vol*kvol*(np.dot(self.A, uni)))/(np.dot(self.h, uni))
                keq = lamb_w_vol*kvol

            q = q + keq*grad_p*(np.dot(self.A, uni))
            kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

        return q

    def div_upwind_3(self, volume, p_tag):

        """
        calcula o fluxo total que entra no volume para calcular a saturacao
        a mobilidade da interface é dada pela media das mobilidades
        """
        qt = 0.0
        qp = 0.0
        q = 0.0
        qw = 0.0
        list_sat = []
        list_lbw = []
        list_gid = []
        list_grad = []
        list_q = []
        list_p = []
        list_lbeq = []

        pvol = self.mb.tag_get_data(p_tag, volume)[0][0]
        adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        volume_centroid = self.mesh_topo_util.get_average_position([volume])
        global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        sat_volume = self.mb.tag_get_data(self.sat_tag, volume, flat=True)[0]
        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
        lbt_vol = self.mb.tag_get_data(self.lbt_tag, volume)[0][0]
        fw_vol = self.mb.tag_get_data(self.fw_tag, volume)[0][0]

        for adj in adjs_vol:
            global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
            sat_adj = self.mb.tag_get_data(self.sat_tag, adj, flat=True)[0]
            padj = self.mb.tag_get_data(p_tag, adj)[0][0]
            lbt_adj = self.mb.tag_get_data(self.lbt_tag, adj)[0][0]
            adj_centroid = self.mesh_topo_util.get_average_position([adj])
            direction = adj_centroid - volume_centroid
            uni = self.unitary(direction)
            kvol = np.dot(np.dot(kvol,uni),uni)
            kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
            kadj = np.dot(np.dot(kadj,uni),uni)
            lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
            keq = self.kequiv(kvol, kadj)
            # if global_adj > global_volume:
            #     grad_p = (padj - pvol)/float(np.dot(self.h, uni))
            # else:
            #     grad_p = (pvol - padj)/float(np.dot(self.h, uni))
            grad_p = (padj - pvol)/float(np.dot(self.h, uni))
            lamb_eq = (lamb_w_vol + lamb_w_adj)/2.0
            keq = keq*lamb_eq
            q = q + keq*(grad_p)*(np.dot(self.A, uni))

            # producao de oleo
            if global_volume in self.wells_prod:
                kvol2 = kvol*(lbt_vol)
                kadj2 = kadj*(lbt_adj)
                keq2 = self.kequiv(kvol2, kadj2)
                qt += grad_p*(keq2)*(np.dot(self.A, uni)) #fluxo total que entra no volume

            list_sat.append(sat_adj)
            list_lbw.append(lamb_w_adj)
            list_gid.append(global_adj)
            list_grad.append(grad_p)
            list_q.append(keq*(grad_p)*(np.dot(self.A, uni)))
            list_p.append(padj)
            list_lbeq.append(lamb_eq)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

        if global_volume in self.wells_prod:
            qp += (1 - fw_vol)*qt # fluxo de oleo que sai do volume
            qw += (fw_vol)*qt #fluxo de agua que sai do volume
            q = q - qw
            self.mb.tag_set_data(self.prod_tag, volume, qp)


        list_sat.append(sat_volume)
        list_lbw.append(lamb_w_vol)
        list_gid.append(global_volume)
        list_q.append(q)
        list_p.append(pvol)


        if q < 0:
            print('divergente upwind de agua menor que zero na funcao div_upwind_3')
            import pdb; pdb.set_trace()

        return q

    def erro(self):
        for volume in self.all_fine_vols:
            if volume in self.wells_d:
                erro = 0.0
                self.mb.tag_set_data(self.err_tag, volume, erro)
                continue

            Pf = self.mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            Pms = self.mb.tag_get_data(self.pms_tag, volume, flat = True)[0]
            erro = abs((Pf - Pms)/float(Pf))
            self.mb.tag_set_data(self.err_tag, volume, erro)

    def erro_2(self):
        for volume in self.all_fine_vols:
            if volume in self.wells_d:
                erro = 0.0
                self.mb.tag_set_data(self.err_tag, volume, erro)
                self.mb.tag_set_data(self.err2_tag, volume, erro)
                continue

            Pf = self.mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            Pms = self.mb.tag_get_data(self.pms_tag, volume, flat = True)[0]
            erro_2 = abs(Pf - Pms)#/float(abs(Pf))
            self.mb.tag_set_data(self.err2_tag, volume, erro_2)
            erro = 100*abs((Pf - Pms)/float(Pf))
            self.mb.tag_set_data(self.err_tag, volume, erro)


    def get_volumes_in_interfaces(self, fine_elems_in_primal, primal_id, **options):

        """
        obtem uma lista com os elementos dos primais adjacentes que estao na interface do primal corrente
        (primal_id)

        se flag == 1 alem dos volumes na interface dos primais adjacentes (volumes_in_interface)
        retorna tambem os volumes no primal corrente que estao na sua interface (volumes_in_primal)

        se flag == 2 retorna apenas os volumes do primal corrente que estao na sua interface (volumes_in_primal)

        """
        #0
        volumes_in_primal = []
        volumes_in_interface = []
        # gids_in_primal = self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True)
        for volume in fine_elems_in_primal:
            #1
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            adjs_volume = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            for adj in adjs_volume:
                #2
                fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, adj, flat=True)
                primal_adj = self.mb.tag_get_data(
                    self.primal_id_tag, int(fin_prim), flat=True)[0]
                if primal_adj != primal_id:
                    #3
                    volumes_in_interface.append(adj)
                    volumes_in_primal.append(volume)
        #0
        volumes_in_primal = list(set(volumes_in_primal))
        if options.get("flag") == 1:
            #1
            return volumes_in_interface, volumes_in_primal
        #0
        elif options.get("flag") == 2:
            #1
            return volumes_in_primal
        #0
        else:
            #1
            return volumes_in_interface

    def get_wells(self):
        """
        obtem os gids dos volumes dos pocos

        wells_d = gids do poco com pressao prescrita
        wells_n = gids do poco com vazao prescrita
        set_p = valor da pressao
        set_q = valor da vazao
        wells_inj = gids dos pocos injetores
        wells_prod = gids dos pocos produtores
        """
        wells_d = []
        wells_n = []
        set_p = []
        set_q = []
        wells_inj = []
        wells_prod = []

        wells_set = self.mb.tag_get_data(self.wells_tag, 0, flat=True)[0]
        self.wells = self.mb.get_entities_by_handle(wells_set)

        for well in self.wells:
            global_id = self.mb.tag_get_data(self.global_id_tag, well, flat=True)[0]
            valor_da_prescricao = self.mb.tag_get_data(self.valor_da_prescricao_tag, well, flat=True)[0]
            tipo_de_prescricao = self.mb.tag_get_data(self.tipo_de_prescricao_tag, well, flat=True)[0]
            #raio_do_poco = mb.tag_get_data(raio_do_poco_tag, well, flat=True)[0]
            tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, well, flat=True)[0]
            #tipo_de_fluido = mb.tag_get_data(tipo_de_fluido_tag, well, flat=True)[0]
            #pwf = mb.tag_get_data(pwf_tag, well, flat=True)[0]
            if tipo_de_prescricao == 0:
                wells_d.append(well)
                set_p.append(valor_da_prescricao)
            else:
                wells_n.append(well)
                set_q.append(valor_da_prescricao)
            if tipo_de_poco == 1:
                wells_inj.append(well)
            else:
                wells_prod.append(well)

        self.wells_d = wells_d
        self.wells_n = wells_n
        self.set_p = set_p
        self.set_q = set_q
        self.wells_inj = wells_inj
        self.wells_prod = wells_prod

    def kequiv(self, k1, k2):
        #keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
        keq = (2*k1*k2)/(k1+k2)

        return keq

    def modificar_matriz(self, A, rows, columns):
        """
        realoca a matriz para o tamanho de linhas 'rows' e colunas 'columns'
        """

        row_map = Epetra.Map(rows, 0, self.comm)
        col_map = Epetra.Map(columns, 0, self.comm)

        C = Epetra.CrsMatrix(Epetra.Copy, row_map, col_map, 3)

        for i in range(rows):
            p = A.ExtractGlobalRowCopy(i)
            values = p[0]
            index_columns = p[1]
            C.InsertGlobalValues(i, values, index_columns)

        C.FillComplete()

        return C

    def modificar_vetor(self, v, nc):
        """
        realoca o tamanho do vetor 'v' para o tamanho 'nc'
        """

        std_map = Epetra.Map(nc, 0, self.comm)
        x = Epetra.Vector(std_map)

        for i in range(nc):
            x[i] = v[i]


        return x

    def mount_lines_1(self, volume, map_id):
        """
        monta as linhas da matriz
        retorna o valor temp_k e o mapeamento temp_id
        map_id = mapeamento dos elementos
        """
        #0
        # volume_centroid = self.mb.tag_get_data(self.centroid_tag, volume, flat=True)
        volume_centroid = self.mesh_topo_util.get_average_position([volume])
        adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume, flat=True)[0]
        lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume, flat=True)[0]
        temp_ids = []
        temp_k = []
        for adj in adj_volumes:
            #1
            # adj_centroid = self.mb.tag_get_data(self.centroid_tag, adj, flat=True)
            adj_centroid = self.mesh_topo_util.get_average_position([adj])
            direction = adj_centroid - volume_centroid
            uni = self.unitary(direction)
            kvol = np.dot(np.dot(kvol,uni),uni)
            kvol = kvol*(lamb_w_vol + lamb_o_vol)
            kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
            lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj, flat=True)[0]
            lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj, flat=True)[0]
            kadj = np.dot(np.dot(kadj,uni),uni)
            kadj = kadj*(lamb_w_adj + lamb_o_adj)
            keq = self.kequiv(kvol, kadj)
            keq = keq*(np.dot(self.A, uni))/float(abs(np.dot(direction, uni)))
            temp_ids.append(map_id[adj])
            temp_k.append(-keq)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        #0
        temp_k.append(-sum(temp_k))
        temp_ids.append(map_id[volume])

        return temp_k, temp_ids

    def multimat_vector(self, A, row, b):
        """
        multiplica a matriz A pelo vetor 'b', 'row' é o numero de linhas de A ou tamanho de b
        """

        std_map = Epetra.Map(row, 0, self.comm)
        c = Epetra.Vector(std_map)

        A.Multiply(False, b, c)

        return c

    def Neuman_problem_4(self):

        colocation_points = mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, self.collocation_point_tag, np.array([None]))

        sets = []
        for col in colocation_points:
            #col = mb.get_entities_by_handle(col)[0]
            sets.append(self.mb.get_entities_by_handle(col)[0])
        sets = set(sets)

        for primal in self.primals:

            volumes_in_interface = []#v1
            volumes_in_primal = []#v2
            primal_id = mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = mb.get_entities_by_handle(primal)
            #setfine_elems_in_primal = set(fine_elems_in_primal)

            for fine_elem in fine_elems_in_primal:

                global_volume = mb.tag_get_data(self.global_id_tag, fine_elem, flat=True)[0]
                volumes_in_primal.append(fine_elem)
                adj_fine_elems = mesh_topo_util.get_bridge_adjacencies(fine_elem, 2, 3)

                for adj in adj_fine_elems:
                    fin_prim = mb.tag_get_data(self.fine_to_primal_tag, adj, flat=True)
                    primal_adj = mb.tag_get_data(
                        self.primal_id_tag, int(fin_prim), flat=True)[0]

                    if primal_adj != primal_id:
                        volumes_in_interface.append(adj)

            volumes_in_primal.extend(volumes_in_interface)
            id_map = dict(zip(volumes_in_primal, range(len(volumes_in_primal))))
            std_map = Epetra.Map(len(volumes_in_primal), 0, comm)
            b = Epetra.Vector(std_map)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            dim = len(volumes_in_primal)
            b_np = np.zeros(dim)
            A_np = np.zeros((dim, dim))

            for volume in volumes_in_primal:
                global_volume = mb.tag_get_data(self.global_id_tag, volume)[0][0]
                temp_id = []
                temp_k = []
                centroid_volume = mesh_topo_util.get_average_position([volume])
                k_vol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                adj_vol = mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                lamb_w_vol = mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
                lamb_o_vol = mb.tag_get_data(self.lamb_o_tag, volume)[0][0]

                if volume in self.wells:
                    tipo_de_prescricao = mb.tag_get_data(self.tipo_de_prescricao_tag, volume)[0][0]
                    if tipo_de_prescricao == 0:
                        valor_da_prescricao = mb.tag_get_data(self.valor_da_prescricao_tag, volume)[0][0]
                        temp_k.append(1.0)
                        temp_id.append(id_map[volume])
                        b[id_map[volume]] = valor_da_prescricao
                        b_np[id_map[volume]] = valor_da_prescricao

                    else:
                        soma = 0.0
                        for adj in adj_vol:
                            centroid_adj = self.mesh_topo_util.get_average_position([adj])
                            direction = centroid_adj - centroid_volume
                            uni = self.unitary(direction)
                            k_vol = np.dot(np.dot(k_vol,uni),uni)
                            k_vol = k_vol*(lamb_w_vol + lamb_o_vol)
                            k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                            k_adj = np.dot(np.dot(k_adj,uni),uni)
                            lamb_w_adj = mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                            lamb_o_adj = mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                            kadj = kadj*(lamb_w_adj + lamb_o_adj)
                            keq = self.kequiv(k_vol, k_adj)
                            keq = keq*(np.dot(self.A, uni)/(np.dot(self.h, uni)))
                            soma = soma + keq
                            temp_k.append(-keq)
                            temp_id.append(id_map[adj])
                        temp_k.append(soma)
                        temp_id.append(id_map[volume])
                        tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                        valor_da_prescricao = self.mb.tag_get_data(self.valor_da_prescricao_tag, volume)[0][0]
                        if tipo_de_poco == 1:
                            b[id_map[volume]] = valor_da_prescricao
                            b_np[id_map[volume]] = valor_da_prescricao
                        else:
                            b[id_map[volume]] = -valor_da_prescricao
                            b_np[id_map[volume]] = -valor_da_prescricao

                elif volume in sets:
                    temp_k.append(1.0)
                    temp_id.append(id_map[volume])
                    b[id_map[volume]] = self.mb.tag_get_data(self.pms_tag, volume)[0]
                    b_np[id_map[volume]] = self.mb.tag_get_data(self.pms_tag, volume)[0]

                elif volume in volumes_in_interface:
                    for adj in adj_vol:
                        fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, adj, flat=True)
                        primal_adj = self.mb.tag_get_data(
                            self.primal_id_tag, int(fin_prim), flat=True)[0]
                        if primal_adj == primal_id:
                            pms_adj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                            pms_volume = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                            b[id_map[volume]] = pms_volume - pms_adj
                            b_np[id_map[volume]] = pms_volume - pms_adj
                            temp_k.append(1.0)
                            temp_id.append(id_map[volume])
                            temp_k.append(-1.0)
                            temp_id.append(id_map[adj])

                else:
                    soma = 0.0
                    for adj in adj_vol:
                        centroid_adj = self.mesh_topo_util.get_average_position([adj])
                        direction = centroid_adj - centroid_volume
                        uni = self.unitary(direction)
                        k_vol = np.dot(np.dot(k_vol,uni),uni)
                        k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        k_adj = np.dot(np.dot(k_adj,uni),uni)
                        keq = self.kequiv(k_vol, k_adj)
                        keq = keq/(np.dot(self.h2, uni))
                        soma = soma + keq
                        temp_k.append(-keq)
                        temp_id.append(id_map[adj])
                    temp_k.append(soma)
                    temp_id.append(id_map[volume])

                A.InsertGlobalValues(id_map[volume], temp_k, temp_id)
                A_np[id_map[volume], temp_id] = temp_k[:]

            A.FillComplete()
            x = self.solve_linear_problem(A, b, dim)
            x_np = np.linalg.solve(A_np, b_np)

            for i in range(len(volumes_in_primal) - len(volumes_in_interface)):
                volume = volumes_in_primal[i]
                self.mb.tag_set_data(self.p_tag, volume, x[i])
                self.mb.tag_set_data(self.pms2_tag, volume, x_np[i])

    def Neuman_problem_4_3(self):
        """
        recalcula as pressoes em cada primal usando fluxo prescrito nas interfaces do primal
        """
        #0
        colocation_points = self.mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, self.collocation_point_tag, np.array([None]))
        sets = []
        for col in colocation_points:
            #1
            #col = mb.get_entities_by_handle(col)[0]
            sets.append(self.mb.get_entities_by_handle(col)[0])
        #0
        sets = set(sets)
        for primal in self.primals:
            #1
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_interface = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id)
            all_volumes = list(fine_elems_in_primal) + volumes_in_interface
            all_volumes_ic = self.all_fine_vols_ic & set(all_volumes)
            gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, all_volumes_ic, flat=True)
            map_volumes = dict(zip(gids_vols_ic, range(len(gids_vols_ic))))
            std_map = Epetra.Map(len(all_volumes_ic), 0, self.comm)
            b = Epetra.Vector(std_map)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            dim = len(all_volumes_ic)
            b_np = np.zeros(dim)
            A_np = np.zeros((dim, dim))
            for volume in all_volumes_ic:
                #2
                soma = 0
                temp_id = []
                temp_k = []
                volume_centroid = self.mesh_topo_util.get_average_position([volume])
                adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
                lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
                if volume in sets:
                    #3
                    temp_k.append(1.0)
                    temp_id.append(map_volumes[global_volume])
                    b[map_volumes[global_volume]] = self.mb.tag_get_data(self.pms_tag, volume)[0]
                    b_np[map_volumes[global_volume]] = self.mb.tag_get_data(self.pms_tag, volume)[0]
                #2
                elif volume in volumes_in_interface:
                    #3
                    for adj in adj_volumes:
                        #4
                        if adj in fine_elems_in_primal:
                            #5
                            global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                            pms_adj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                            pms_volume = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                            b[map_volumes[global_volume]] = pms_volume - pms_adj
                            b_np[map_volumes[global_volume]] = pms_volume - pms_adj
                            temp_k.append(1.0)
                            temp_id.append(map_volumes[global_volume])
                            temp_k.append(-1.0)
                            temp_id.append(map_volumes[global_adj])
                        #4
                        else:
                            #5
                            pass
                #2
                else:
                    #3
                    for adj in adj_volumes:
                        #4
                        global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                        adj_centroid = self.mesh_topo_util.get_average_position([adj])
                        direction = adj_centroid - volume_centroid
                        uni = self.unitary(direction)
                        kvol = np.dot(np.dot(kvol,uni),uni)
                        kvol = kvol*(lamb_w_vol + lamb_o_vol)
                        kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        kadj = np.dot(np.dot(kadj,uni),uni)
                        lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                        lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                        kadj = kadj*(lamb_w_adj + lamb_o_adj)
                        keq = self.kequiv(kvol, kadj)
                        keq = keq*(np.dot(self.A, uni))/(np.dot(self.h, uni))
                        soma = soma + keq
                        if global_adj in self.wells_d:
                            #5
                            index = self.wells_d.index(global_adj)
                            b[map_volumes[global_volume]] += self.set_p[index]*(keq)
                            b_np[map_volumes[global_volume]] += self.set_p[index]*(keq)
                        #4
                        else:
                            #5
                            temp_id.append(map_volumes[global_adj])
                            temp_k.append(-keq)
                        #4
                        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    #3
                    temp_k.append(soma)
                    temp_id.append(map_volumes[global_volume])
                    if global_volume in self.wells_n:
                        #4
                        index = self.wells_n.index(global_volume)
                        tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)[0]
                        if tipo_de_poco == 1:
                            #5
                            b[map_volumes[global_volume]] += self.set_q[index]
                            b_np[map_volumes[global_volume]] += self.set_q[index]
                        #4
                        else:
                            #5
                            b[map_volumes[global_volume]] += -self.set_q[index]
                            b_np[map_volumes[global_volume]] += -self.set_q[index]
                #2
                A.InsertGlobalValues(map_volumes[global_volume], temp_k, temp_id)
                A_np[map_volumes[global_volume], temp_id] = temp_k
            #1
            A.FillComplete()
            x = self.solve_linear_problem(A, b, dim)
            x_np = np.linalg.solve(A_np, b_np)
            for volume in all_volumes_ic:
                #2
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                self.mb.tag_set_data(self.pcorr_tag, volume, x[map_volumes[global_volume]])
                self.mb.tag_set_data(self.pms2_tag, volume, x_np[map_volumes[global_volume]])
            #1
            for volume in set(all_volumes) - all_volumes_ic:
                #2
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                index = self.wells_d.index(global_volume)
                p = self.set_p[index]
                self.mb.tag_set_data(self.pcorr_tag, volume, p)
                self.mb.tag_set_data(self.pms2_tag, volume, p)

    def Neuman_problem_6(self):
        # self.set_of_collocation_points_elems = set()
        #0
        """
        map_volumes[volume]
        map_volumes[adj]
        """
        for primal in self.primals:
            #1
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_interface, volumes_in_primal = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id, flag = 1)
            all_volumes = list(fine_elems_in_primal)
            dim = len(all_volumes)
            map_volumes = dict(zip(all_volumes, range(len(all_volumes))))
            std_map = Epetra.Map(len(all_volumes), 0, self.comm)
            b = Epetra.Vector(std_map)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            # b_np = np.zeros(dim)
            # A_np = np.zeros((dim, dim))
            for volume in all_volumes:
                #2
                # import pdb; pdb.set_trace()
                soma = 0
                temp_k = []
                temp_id = []
                gid1 = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                centroid_volume = self.mesh_topo_util.get_average_position([volume])
                k_vol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
                lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                # print('in wells d: {0}'.format(volume in self.wells_d))
                # print('in collocation_points: {0}'.format(volume in self.set_of_collocation_points_elems))
                # print('in volumes_in_primal: {0}'.format(volume in volumes_in_primal))
                # import pdb; pdb.set_trace()
                if volume in self.wells_d or volume in self.set_of_collocation_points_elems:
                    #3
                    value = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    temp_k.append(1.0)
                    temp_id.append(map_volumes[volume])
                    b[map_volumes[volume]] = value
                    #b_np[map_volumes[volume]] = value
                #2
                elif volume in volumes_in_primal:
                    #3
                    for adj in adjs_vol:
                        #4
                        gid2 = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                        padj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                        centroid_adj = self.mesh_topo_util.get_average_position([adj])
                        direction = centroid_adj - centroid_volume
                        uni = self.unitary(direction)
                        # h = abs(np.dot(direction, uni))
                        k_vol = np.dot(np.dot(k_vol,uni),uni)
                        k_vol = k_vol*(lamb_w_vol + lamb_o_vol)
                        k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                        lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                        k_adj = np.dot(np.dot(k_adj,uni),uni)
                        k_adj = k_adj*(lamb_w_adj + lamb_o_adj)
                        keq = self.kequiv(k_vol, k_adj)
                        keq = keq*(np.dot(self.A, uni)/np.dot(self.h, uni))
                        if adj in all_volumes:
                            #5
                            soma += keq
                            temp_k.append(-keq)
                            temp_id.append(map_volumes[adj])
                        #4
                        else:
                            #5
                            q_in = (padj - pvol)*(keq)
                            # print('qin: {0}'.format(q_in))
                            # print('gidvol: {0}; gidadj: {1}'.format(gid1, gid2))
                            # print('pvol: {0}; padj: {1}'.format(pvol, padj))
                            # print('keq: {0}\n'.format(keq))
                            # import pdb; pdb.set_trace()
                            b[map_volumes[volume]] += q_in
                            #b_np[map_volumes[volume]] += q_in
                        k_vol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    #3
                    temp_k.append(-sum(temp_k))
                    temp_id.append(map_volumes[volume])
                    if volume in self.wells_n:
                        #4
                        index = self.wells_n.index(volume)
                        if volume in self.wells_inj:
                            #5
                            b[map_volumes[volume]] += self.set_q[index]
                            #b_np[map_volumes[volume]] += self.set_q[index]
                        #4
                        else:
                            #5
                            b[map_volumes[volume]] -= self.set_q[index]
                            #b_np[map_volumes[volume]] -= self.set_q[index]
                #2
                else:
                    #3
                    temp_k, temp_id = self.mount_lines_1(volume, map_volumes)
                    if volume in self.wells_n:
                        #4
                        index = self.wells_n.index(volume)
                        if volume in self.wells_inj:
                            #5
                            b[map_volumes[volume]] += self.set_q[index]
                            #b_np[map_volumes[volume]] += self.set_q[index]
                        #4
                        else:
                            #5
                            b[map_volumes[volume]] -= self.set_q[index]
                            #b_np[map_volumes[volume]] -= self.set_q[index]
                #2
                A.InsertGlobalValues(map_volumes[volume], temp_k, temp_id)
                #A_np[map_volumes[volume], temp_id] = temp_k
                # print('primal_id')
                # print(self.ident_primal[primal_id])
                # print('gid: {0}'.format(gid1))
                # print('temp_id:{0}'.format(temp_id))
                # print('temp_k:{0}'.format(temp_k))
                # print(A_np[map_volumes[volume]])
                # print('b_np:{0}'.format(b_np[map_volumes[volume]]))
            #1
            A.FillComplete()
            x = self.solve_linear_problem(A, b, dim)
            #x_np = np.linalg.solve(A_np, b_np)
            # print(x_np)
            for volume in all_volumes:
                #2
                gid1 = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                self.mb.tag_set_data(self.pcorr_tag, volume, x[map_volumes[volume]])
                #self.mb.tag_set_data(self.pms2_tag, volume, x_np[map_volumes[volume]])

    def organize_op(self):
        """
        elimina as linhas do operador de prolongamento que se referem aos volumes com pressao prescrita
        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic), 0, self.comm)
        trilOP2 = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
        gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols_ic, flat=True)
        cont = 0
        for elem in self.all_fine_vols_ic:
            #1
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            p = self.trilOP.ExtractGlobalRowCopy(gid)
            values = p[0]
            index = p[1]
            trilOP2.InsertGlobalValues(self.map_vols_ic[elem], list(values), list(index))
        #0
        self.trilOP = trilOP2
        self.trilOP.FillComplete()

    def organize_Pf(self):

        """
        organiza a solucao da malha fina para setar no arquivo de saida
        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)
        Pf2 = Epetra.Vector(std_map)
        for i in range(len(self.Pf)):
            #1
            value = self.Pf[i]
            elem = self.map_vols_ic_2[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pf2[gid] = value
        #0
        for i in range(len(self.wells_d)):
            #1
            value = self.set_p[i]
            elem = self.wells_d[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pf2[gid] = value
        #0
        self.Pf_all = Pf2

    def organize_Pms(self):

        """
        organiza a solucao do Pms para setar no arquivo de saida
        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)
        Pms2 = Epetra.Vector(std_map)
        for i in range(len(self.Pms)):
            #1
            value = self.Pms[i]
            elem = self.map_vols_ic_2[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pms2[gid] = value
        #0
        for i in range(len(self.wells_d)):
            #1
            value = self.set_p[i]
            elem = self.wells_d[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pms2[gid] = value
        #0
        self.Pms_all = Pms2

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

        if y == self.krw_r:
            if S <= 0.2:
                pol = 0.0
            else:
                pass
        elif y == self.kro_r:
            if S <= 0:
                pol = 1.0
            elif S >= 0.9:
                pol = 0.0
            else:
                pass
        else:
            pass

        return abs(pol)

    def pol_interp_2(self, S):

        # S_temp = (S - self.Swc)/(1 - self.Swc - self.Sor)
        # krw = (S_temp)**(self.nw)
        # kro = (1 - S_temp)**(self.no)
        if S > (1 - self.Sor):
            krw = 1.0
            kro = 0.0
        elif S < self.Swc:
            krw = 0.0
            kro = 1.0
        else:
            krw = ((S - self.Swc)/float(1 - self.Swc - self.Sor))**(self.nw)
            kro = ((1 - S - self.Swc)/float(1 - self.Swc - self.Sor))**(self.no)

        return krw, kro

    def pymultimat(self, A, B, nf):
        """
        multiplica a matriz A pela B
        """

        nf_map = Epetra.Map(nf, 0, self.comm)

        C = Epetra.CrsMatrix(Epetra.Copy, nf_map, 3)

        EpetraExt.Multiply(A, False, B, False, C)

        C.FillComplete()

        return C

    def read_perm_rel(self):
        """
        le o arquivo perm_rel.py para usar na funcao pol_interp
        """
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

    def read_perms_and_phi_spe10(self):
        nx = 60
        ny = 220
        nz = 85
        N = nx*ny*nz
        # l1 = [N, 2*N, 3*N]
        # l2 = [0, 1, 2]
        #
        # ks = np.loadtxt('spe_perm.dat')
        # t1, t2 = ks.shape
        # ks = ks.reshape((t1*t2))
        # ks2 = np.zeros((N, 9))
        #
        #
        # for i in range(0, N):
        #     # as unidades do spe_10 estao em milidarcy
        #     # unidade de darcy em metro quadrado =  (1 Darcy)*(9.869233e-13 m^2/Darcy)
        #     # fonte -- http://www.calculator.org/property.aspx?name=permeability
        #     ks2[i, 0] = ks[i]*(10**(-3))# *9.869233e-13
        #
        # cont = 0
        # for i in range(N, 2*N):
        #     ks2[cont, 4] = ks[i]*(10**(-3))# *9.869233e-13
        #     cont += 1
        #
        # cont = 0
        # for i in range(2*N, 3*N):
        #     ks2[cont, 8] = ks[i]*(10**(-3))# *9.869233e-13
        #     cont += 1
        #
        #
        #
        # cont = None
        # phi = np.loadtxt('spe_phi.dat')
        # t1, t2 = phi.shape
        # phi = phi.reshape(t1*t2)
        # np.savez_compressed('spe10_perms_and_phi', perms = ks2, phi = phi)
        # ks2 = None
        #
        # # obter a permeabilidade de uma regiao
        # # digitar o inicio e o fim da regiao

        ks = np.load('spe10_perms_and_phi.npz')['perms']
        phi = np.load('spe10_perms_and_phi.npz')['phi']


        gid1 = [0, 0, 50]
        gid2 = [gid1[0] + self.nx-1, gid1[1] + self.ny-1, gid1[2] + self.nz-1]

        gid1 = np.array(gid1)
        gid2 = np.array(gid2)

        dif = gid2 - gid1 + np.array([1, 1, 1])
        permeabilidade = []
        fi = []

        cont = 0
        for k in range(dif[2]):
            for j in range(dif[1]):
                for i in range(dif[0]):
                    gid = gid1 + np.array([i, j, k])
                    gid = gid[0] + gid[1]*nx + gid[2]*nx*ny
                    # permeabilidade[cont] = ks[gid]
                    permeabilidade.append(ks[gid])
                    fi.append(phi[gid])
                    cont += 1
        cont = 0

        for volume in self.all_fine_vols:
            self.mb.tag_set_data(self.perm_tag, volume, permeabilidade[cont])
            self.mb.tag_set_data(self.fi_tag, volume, fi[cont])
            cont += 1



        # self.mb.tag_set_data(self.perm_tag, self.all_fine_vols, permeabilidade)
        # self.mb.tag_set_data(self.fi_tag, self.all_fine_vols, fi)
        for volume in self.all_fine_vols:
            gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)
            perm = self.mb.tag_get_data(self.perm_tag, volume).reshape([3,3])
            fi2 = self.mb.tag_get_data(self.fi_tag, volume, flat = True)[0]


    def read_structured(self):
        with open('structured.cfg', 'r') as arq:
            text = arq.readlines()

        a = text[11].strip()
        a = a.split("=")
        a = a[1].strip()
        a = a.split(",")
        crx = int(a[0].strip())
        cry = int(a[1].strip())
        crz = int(a[2].strip())

        a = text[12].strip()
        a = a.split("=")
        a = a[1].strip()
        a = a.split(",")
        nx = int(a[0].strip())
        ny = int(a[1].strip())
        nz = int(a[2].strip())

        a = text[13].strip()
        a = a.split("=")
        a = a[1].strip()
        a = a.split(",")
        hx = float(a[0].strip())
        hy = float(a[1].strip())
        hz = float(a[2].strip())

        tx = hx*nx
        ty = hy*ny
        tz = hz*nz
        h = np.array([hx, hy, hz])
        h2 = np.array([hx**2, hy**2, hz**2])

        ax = hy*hz
        ay = hx*hz
        az = hx*hy
        a = np.array([ax, ay, az])

        hmin = min(hx, hy, hz)
        V = hx*hy*hz

        self.nx = nx # numero de volumes na direcao x
        self.ny = ny # numero de volumes na direcao y
        self.nz = nz # numero de volumes na direcao z
        self.h2 = h2 # vetor com os tamanhos ao quadrado de cada volume
        self.h = h # vetor com os tamanhos de cada volume
        self.V = V # volume de um volume da malha fina
        self.A = a # vetor com as areas
        self.tz = tz # tamanho total na direcao z
        self.viz_x = [1, -1]
        self.viz_y = [nx, -nx]
        self.viz_z = [nx*ny, -nx*ny]

    def set_erro(self):
        """
        modulo da diferenca entre a pressao da malha fina e a multiescala
        """
        for volume in self.all_fine_vols:
            Pf = mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            Pms = mb.tag_get_data(self.pms_tag, volume, flat = True)[0]
            erro = abs(Pf - Pms)/float(abs(Pf))
            mb.tag_set_data(self.err_tag, volume, erro)

    def set_fi(self):
        fi = 0.3
        for volume in self.all_fine_vols:
            self.mb.tag_set_data(self.fi_tag, volume, fi)

    def set_global_problem(self):

        std_map = Epetra.Map(len(self.all_fine_vols), 0, comm)

        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)

        for volume in self.all_fine_vols:

            volume_centroid = mesh_topo_util.get_average_position([volume])
            adj_volumes = mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            lamb_w_vol = mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
            global_volume = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]

            if global_volume not in self.wells_d:

                soma = 0.0
                temp_glob_adj = []
                temp_k = []

                for adj in adj_volumes:
                    global_adj = mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    adj_centroid = mesh_topo_util.get_average_position([adj])

                    direction = adj_centroid - volume_centroid
                    uni = self.unitary(direction)
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kvol = kvol*(lamb_w_vol + lamb_o_vol)
                    kadj = mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    lamb_w_adj = mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                    lamb_o_adj = mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                    kadj = kadj*(lamb_w_adj + lamb_o_adj)
                    keq = self.kequiv(kvol, kadj)
                    keq = keq/(np.dot(self.h2, uni))
                    temp_glob_adj.append(global_adj)
                    temp_k.append(keq)
                    soma = soma + keq
                    kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

                soma = -1*soma
                temp_k.append(soma)
                temp_glob_adj.append(global_volume)
                #print(temp_k)
                #print(temp_glob_adj)
                self.trans_fine.InsertGlobalValues(global_volume, temp_k, temp_glob_adj)

                if global_volume in self.wells_n:
                    index = self.wells_n.index(global_volume)
                    tipo_de_poco = mb.tag_get_data(self.tipo_de_poco_tag, volume)
                    if tipo_de_poco == 1:
                        self.b[global_volume] = -self.set_q[index]
                    else:
                        self.b[global_volume] = self.set_q[index]

            else:
                index = self.wells_d.index(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, [1.0], [global_volume])
                self.b[global_volume] = self.set_p[index]

        self.trans_fine.FillComplete()

    def set_global_problem_gr_vf(self):

        """
        transmissibilidade da malha fina com gravidade _vf
        """

        self.gama = 1.0

        std_map = Epetra.Map(len(self.all_fine_vols),0,comm)

        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)

        for volume in self.all_fine_vols:

            volume_centroid = mesh_topo_util.get_average_position([volume])
            adj_volumes = mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            lamb_w_vol = mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
            global_volume = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]

            if global_volume not in self.wells_d:

                soma = 0.0
                soma2 = 0.0
                soma3 = 0.0
                temp_glob_adj = []
                temp_k = []

                for adj in adj_volumes:
                    global_adj = mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    adj_centroid = mesh_topo_util.get_average_position([adj])
                    direction = adj_centroid - volume_centroid
                    altura = adj_centroid[2]
                    uni = self.unitary(direction)
                    z = uni[2]
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kvol = kvol*(lamb_w_vol + lamb_o_vol)
                    kadj = mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    lamb_w_adj = mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                    lamb_o_adj = mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                    kadj = kadj*(lamb_w_adj + lamb_o_adj)
                    keq = self.kequiv(kvol, kadj)
                    keq = keq*(np.dot(self.A, uni))/(np.dot(self.h, uni))

                    if z == 1.0:
                        keq2 = keq*self.gama_
                        soma2 = soma2 + keq2
                        soma3 = soma3 + (-keq2*(self.tz-altura))

                    temp_glob_adj.append(global_adj)
                    temp_k.append(keq)

                    soma = soma + keq

                soma2 = soma2*(self.tz-volume_centroid[2])
                soma2 = -(soma2 + soma3)
                soma = -1*soma
                temp_k.append(soma)
                temp_glob_adj.append(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, temp_k, temp_glob_adj)

                if global_volume in self.wells_n:
                    index = self.wells_n.index(global_volume)
                    tipo_de_poco = mb.tag_get_data(self.tipo_de_poco_tag, volume)[0][0]
                    if tipo_de_poco == 1:
                        self.b[global_volume] = -self.set_q[index] + soma2
                    else:
                        self.b[global_volume] = self.set_q[index] + soma2
                else:
                    self.b[global_volume] = soma2

                kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

            else:
                index = self.wells_d.index(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, [1.0], [global_volume])
                self.b[global_volume] = self.set_p[index]

        self.trans_fine.FillComplete()

    def set_global_problem_vf(self):

        std_map = Epetra.Map(len(self.all_fine_vols),0, comm)

        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)

        for volume in self.all_fine_vols:

            volume_centroid = mesh_topo_util.get_average_position([volume])
            adj_volumes = mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            lamb_w_vol = mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
            global_volume = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]

            if global_volume not in self.wells_d:

                soma = 0.0
                temp_glob_adj = []
                temp_k = []

                for adj in adj_volumes:

                    global_adj = mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    adj_centroid = mesh_topo_util.get_average_position([adj])
                    direction = adj_centroid - volume_centroid
                    uni = self.unitary(direction)
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kvol = kvol*(lamb_w_vol + lamb_o_vol)
                    kadj = mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    lamb_w_adj = mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                    lamb_o_adj = mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                    kadj = kadj*(lamb_w_adj + lamb_o_adj)
                    keq = self.kequiv(kvol, kadj)
                    keq = keq*(np.dot(self.A, uni)/(np.dot(self.h, uni)))
                    temp_glob_adj.append(global_adj)
                    temp_k.append(keq)
                    soma = soma + keq
                    kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

                soma = -1*soma
                temp_k.append(soma)
                temp_glob_adj.append(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, temp_k, temp_glob_adj)

                if global_volume in self.wells_n:
                    index = self.wells_n.index(global_volume)
                    tipo_de_poco = mb.tag_get_data(self.tipo_de_poco_tag, volume)
                    if tipo_de_poco == 1:
                        self.b[global_volume] = -self.set_q[index]
                    else:
                        self.b[global_volume] = self.set_q[index]

            else:
                index = self.wells_d.index(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, [1.0], [global_volume])
                self.b[global_volume] = self.set_p[index]

        self.trans_fine.FillComplete()

        """for i in range(self.nf):
            p = self.trans_fine.ExtractGlobalRowCopy(i)
            print(p[0])
            print(p[1])
            print('soma')
            print(sum(p[0]))
            if abs(sum(p[0])) > 0.000001 and abs(sum(p[0])) != 1.0:
                print('Erroooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
            print('\n')"""

    def set_global_problem_vf_2(self):
        """
        transmissibilidade da malha fina excluindo os volumes com pressao prescrita
        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)
        for volume in self.all_fine_vols_ic - set(self.neigh_wells_d):
            #1
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
            soma = 0.0
            temp_glob_adj = []
            temp_k = []
            for adj in adj_volumes:
                #2
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kvol = kvol*(lamb_w_vol + lamb_o_vol)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                kadj = kadj*(lamb_w_adj + lamb_o_adj)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni)/(np.dot(self.h, uni)))
                temp_glob_adj.append(self.map_vols_ic[adj])
                temp_k.append(-keq)
                soma = soma + keq
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            temp_k.append(soma)
            temp_glob_adj.append(self.map_vols_ic[volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                index = self.wells_n.index(volume)
                if volume in self.wells_inj:
                    #3
                    self.b[self.map_vols_ic[volume]] += self.set_q[index]
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[volume]] += -self.set_q[index]
        #0
        for volume in self.neigh_wells_d:
            #1
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
            soma = 0.0
            temp_glob_adj = []
            temp_k = []
            for adj in adj_volumes:
                #2
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kvol = kvol*(lamb_w_vol + lamb_o_vol)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                kadj = kadj*(lamb_w_adj + lamb_o_adj)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni)/(np.dot(self.h, uni)))
                if adj in self.wells_d:
                    #3
                    soma = soma + keq
                    index = self.wells_d.index(adj)
                    self.b[self.map_vols_ic[volume]] += self.set_p[index]*(keq)
                #2
                else:
                    #3
                    temp_glob_adj.append(self.map_vols_ic[adj])
                    temp_k.append(-keq)
                    soma = soma + keq
                #2
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            temp_k.append(soma)
            temp_glob_adj.append(self.map_vols_ic[volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                index = self.wells_n.index(volume)
                if volume in self.wells_inj:
                    #3
                    self.b[self.map_vols_ic[volume]] += self.set_q[index]
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[volume]] += -self.set_q[index]
        #0
        self.trans_fine.FillComplete()

    def set_k(self):
        """
        seta as permeabilidades dos volumes
        """


        # perm_tensor = [1, 0.0, 0.0,
        #                 0.0, 1, 0.0,
        #                 0.0, 0.0, 1]
        #
        # for volume in self.all_fine_vols:
        #     self.mb.tag_set_data(self.perm_tag, volume, perm_tensor)

        perm_tensor_1 = [1, 0.0, 0.0,
                        0.0, 1, 0.0,
                        0.0, 0.0, 1]

        perm_tensor_2 = [0.5, 0.0, 0.0,
                         0.0, 0.5, 0.0,
                         0.0, 0.0, 0.5]

        gid1 = np.array([0, 0, 0])
        gid2 = np.array([int((self.nx - 1)/2.0), int((self.ny - 1)/2.0), int((self.nz - 1)/2.0)])
        dif = gid2 - gid1 + np.array([1, 1, 1])

        gids = []
        for k in range(dif[2]):
            for j in range(dif[1]):
                for i in range(dif[0]):
                    gid = gid1 + np.array([i, j, k])
                    gid = gid[0] + gid[1]*self.nx + gid[2]*self.nx*self.ny
                    gids.append(gid)


        for volume in self.all_fine_vols:
            gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
            if gid_vol in gids:
                self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_1)
            else:
                self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_2)


        # for volume in self.all_fine_vols:
        #     k = random.randint(1, 10001)*1e-3
        #
        #     perm_tensor = [k, 0.0, 0.0,
        #                    0.0, k, 0.0,
        #                    0.0, 0.0, k]
        #
        #     self.mb.tag_set_data(self.perm_tag, volume, perm_tensor)

        # perm_tensor = [10.0,  0.0, 0.0,
        #                 0.0, 10.0, 0.0,
        #                 0.0,  0.0, 1.0]
        # for volume in self.all_fine_vols:
        #     self.mb.tag_set_data(self.perm_tag, volume, perm_tensor)


        # perm_tensor = [10.0,  0.0, 0.0,
        #                 0.0, 10.0, 0.0,
        #                 0.0,  0.0, 1.0]
        #
        # perm_tensor2 = [20.0,  0.0, 0.0,
        #                  0.0, 20.0, 0.0,
        #                  0.0,  0.0, 2.0]
        #
        # cont = 0
        # for elem in self.all_fine_vols:
        #     if cont%2 == 0:
        #         self.mb.tag_set_data(self.perm_tag, elem, perm_tensor)
        #     else:
        #         self.mb.tag_set_data(self.perm_tag, elem, perm_tensor2)
        #     cont += 1


        # for volume in self.all_fine_vols:
        #     k = random.randint(1, 10001)*1e-3
        #     perm_tensor = [k, 0, 0,
        #                    0, k, 0,
        #                    0, 0, 0.1*k]
        #     # perms.append(perm_tensor)
        #     self.mb.tag_set_data(self.perm_tag, volume, perm_tensor)


        # perm = []
        # for volume in self.all_fine_vols:
        #     k = random.randint(1, 1001)*(10**(-3))
        #     perm_tensor = [k, 0, 0,
        #                    0, k, 0,
        #                    0, 0, k]
        #     perm.append(np.array(perm_tensor))
        #     self.mb.tag_set_data(self.perm_tag, volume, perm_tensor)
        #
        # perm = np.array(perm)
        #
        # np.savez_compressed('perms2', perms = perm)

        # perm = np.load('perms_het.npz')['perms']
        #
        # cont = 0
        # for volume in self.all_fine_vols:
        #     self.mb.tag_set_data(self.perm_tag, volume, perm[cont])
        #     cont += 1
        # # cont = 0


    def set_lamb(self):
        """
        seta o lambda usando pol_interp
        """
        for volume in self.all_fine_vols:
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
            S = self.mb.tag_get_data(self.sat_tag, volume)[0][0]
            krw = self.pol_interp(S, self.Sw_r, self.krw_r)
            kro = self.pol_interp(S, self.Sw_r, self.kro_r)
            lamb_w = krw/self.mi_w
            lamb_o = kro/self.mi_o
            self.mb.tag_set_data(self.lamb_w_tag, volume, lamb_w)
            self.mb.tag_set_data(self.lamb_o_tag, volume, lamb_o)

    def set_lamb_2(self):
        """
        seta o lambda usando pol_interp_2
        """
        for volume in self.all_fine_vols:
            S = self.mb.tag_get_data(self.sat_tag, volume)[0][0]
            krw, kro = self.pol_interp_2(S)
            lamb_w = krw/self.mi_w
            lamb_o = kro/self.mi_o
            lbt = lamb_w + lamb_o
            gid = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
            fw = lamb_w/float(lbt)
            self.mb.tag_set_data(self.lamb_w_tag, volume, lamb_w)
            self.mb.tag_set_data(self.lamb_o_tag, volume, lamb_o)
            self.mb.tag_set_data(self.fw_tag, volume, fw)
            self.mb.tag_set_data(self.lbt_tag, volume, lbt)

    def set_Pc(self):
        """
        seta as pressoes da malha grossa primal
        """

        for primal in self.primals:

            primal_id = mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id]

            fine_elems_in_primal = mb.get_entities_by_handle(primal)
            value = self.Pc[primal_id]
            mb.tag_set_data(
                self.pc_tag,
                fine_elems_in_primal,
                np.repeat(value, len(fine_elems_in_primal)))

    def set_sat_in(self):
        """
        seta a saturacao inicial
        """

        l = []
        for volume in self.wells:
            tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)[0][0]
            if tipo_de_poco == 1:
                gid = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                l.append(gid)


        for volume in self.all_fine_vols:
            gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if gid in l:
                self.mb.tag_set_data(self.sat_tag, volume, 0.8)
            else:
                self.mb.tag_set_data(self.sat_tag, volume, 0.2)

    def set_vel(self, p_tag):

        for volume in self.all_fine_vols_ic:
            v1 = np.zeros(3)
            # v2 = np.zeros(3)
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            front = np.array([global_volume + self.viz_x[0], global_volume + self.viz_y[0], global_volume + self.viz_z[0]])
            back = np.array([global_volume - self.viz_x[0], global_volume - self.viz_y[0], global_volume - self.viz_z[0]])
            viz_x = np.array([global_volume + self.viz_x[0], global_volume - self.viz_x[0]])
            viz_y = np.array([global_volume + self.viz_y[0], global_volume - self.viz_y[0]])
            viz_z = np.array([global_volume + self.viz_z[0], global_volume - self.viz_z[0]])
            lbt_vol = self.mb.tag_get_data(self.lbt_tag, volume)[0][0]
            pvol = self.mb.tag_get_data(self.p_tag, volume)[0][0]
            for adj in adj_volumes:
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                padj = self.mb.tag_get_data(self.p_tag, adj)[0][0]
                lbt_adj = self.mb.tag_get_data(self.lbt_tag, adj)[0][0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kvol = kvol*(lbt_vol)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                kadj = kadj*(lbt_adj)
                keq = self.kequiv(kvol, kadj)
                # keq = keq*(np.dot(self.A, uni)/(np.dot(self.h, uni)))
                grad_p = (padj - pvol)/float(np.dot(self.h, uni))
                vel = -(grad_p)*keq
                # if global_adj in front:
                if global_adj > global_volume:
                    if global_adj in viz_x:
                        v1[0] = vel
                    elif global_adj in viz_y:
                        v1[1] = vel
                    else:
                        v1[2] = vel
                else:
                    # if global_adj in viz_x:
                    #     v2[0] = vel
                    # elif global_adj in viz_y:
                    #     v2[1] = vel
                    # else:
                    #     v2[2] = vel
                    pass
            #1
            self.mb.tag_set_data(self.vel_tag, volume, v1)

    def set_volumes_in_primal(self):

        volumes_in_primal_set = self.mb.create_meshset()

        for primal in self.primals:
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_interface, volumes_in_primal = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id, flag = 1)
            for volume in volumes_in_primal:
                self.mb.add_entities(volumes_in_primal_set, [volume])
        self.mb.tag_set_data(self.volumes_in_primal_tag, 0, volumes_in_primal_set)

        # volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        # volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)
        #
        # for primal in self.primals:
        #     primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
        #     fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
        #     volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
        #     gids = self.mb.tag_get_data(self.global_id_tag, volumes_in_primal, flat=True)
        #
        #     print(gids)
        #     import pdb; pdb.set_trace()

    def solve_linear_problem(self, A, b, n):
        """
        resolve o sistema linear da matriz A e termo fonte b
        """
        std_map = Epetra.Map(n, 0, self.comm)

        x = Epetra.Vector(std_map)

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-9)

        return x

    def solve_linear_problem_numpy(self):

        trans_fine_np = np.zeros((self.nf, self.nf))
        b_np = np.zeros(self.nf)

        for i in range(self.nf):
            p = self.trans_fine.ExtractGlobalRowCopy(i)
            #print(p[0])
            #print(p[1])
            trans_fine_np[i, p[1]] = p[0]
            b_np[i] = self.b[i]


        self.Pf2 = np.linalg.solve(trans_fine_np, b_np)
        mb.tag_set_data(self.pf2_tag, self.all_fine_vols, np.asarray(self.Pf2))

    def test_conservation_coarse(self):
        """
        verifica se o fluxo é conservativo nos volumes da malha grossa
        utilizando a pressao multiescala para calcular os fluxos na interface dos mesmos
        """
        #0
        lim = 1e-5
        soma = 0
        Qc2 = []
        prim = []
        for primal in self.primals:
            #1
            Qc = 0
            my_adjs = set()
            primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id1]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_interface, volumes_in_primal = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id1, flag = 1)
            gids = self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True)
            for volume in volumes_in_primal:
                #2
                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                for adj in adjs_vol:
                    #3
                    if adj not in volumes_in_interface or adj in my_adjs:
                        continue
                    my_adjs.add(adj)
                    gid_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    padj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    centroid_volume = self.mesh_topo_util.get_average_position([volume])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_volume
                    uni = self.unitary(direction)
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
                    lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
                    lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                    lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                    kvol = kvol*(lamb_w_vol + lamb_o_vol)
                    kadj = kadj*(lamb_w_adj + lamb_o_adj)
                    keq = self.kequiv(kvol, kadj)
                    keq = keq*(np.dot(self.A, uni))#*np.dot(self.h, uni))
                    grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                    q = (grad_p)*keq
                    # print(gid_vol)
                    # print(gid_adj)
                    # print(pvol)
                    # print(padj)
                    # print(grad_p)
                    # print(q)
                    # print('\n')
                    # import pdb; pdb.set_trace()
                    Qc += q
            #1
            # print('Primal:{0} ///// Qc: {1}'.format(primal_id, Qc))

            Qc2.append(Qc)
            prim.append(primal_id1)
            # print(Qc2)
            # print(prim)
            # import pdb; pdb.set_trace()
            self.mb.tag_set_data(self.flux_coarse_tag, fine_elems_in_primal, np.repeat(Qc, len(fine_elems_in_primal)))
            # if Qc > lim:
            #     print('Qc nao deu zero')
            #     import pdb; pdb.set_trace()
        with open('Qc_{0}.txt'.format(self.loop), 'w') as arq:
            for i,j in zip(prim, Qc2):
                arq.write('Primal:{0} ///// Qc: {1}\n'.format(i, j))
            arq.write('\n')
            arq.write('sum Qc:{0}'.format(sum(Qc2)))

        if sum(Qc2) > lim:
            print('sum QC: {0}'.format(sum(Qc2)))
            import pdb; pdb.set_trace()


    def unitary(self, l):
        """
        obtem o vetor unitario na direcao positiva de l
        """
        uni = l/np.linalg.norm(l)
        uni = uni*uni

        return uni

    def vel_max(self, p_tag):
        """
        Calcula a velocidade maxima tambem a variacao do fluxo fracionario com a saturacao
        """
        lim = 10**(-10)
        v2 = 0.0
        h2 = 0
        dfds2 = 0
        for volume in self.all_fine_vols:
            v = 0.0
            pvol = self.mb.tag_get_data(p_tag, volume)[0][0]
            adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
            sat_vol = self.mb.tag_get_data(self.sat_tag, volume)[0][0]
            for adj in adjs_vol:
                padj = self.mb.tag_get_data(p_tag, adj)[0][0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kvol = kvol*(lamb_w_vol + lamb_o_vol)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                kadj = kadj*(lamb_w_adj + lamb_o_adj)
                keq = self.kequiv(kvol, kadj)
                h = (np.dot(self.h, uni))
                keq = keq/h
                sat_adj = self.mb.tag_get_data(self.sat_tag, adj)[0][0]
                if abs(sat_adj - sat_vol) < lim:
                    continue
                dfds = ((lamb_w_adj/(lamb_w_adj+lamb_o_adj)) - (lamb_w_vol/(lamb_w_vol+lamb_o_vol)))/float((sat_adj - sat_vol))
                v = abs(keq*(padj - pvol)/float(h))
                if v > v2:
                    v2 = v
                    h2 = h
                if abs(dfds) > dfds2:
                    dfds2 = abs(dfds)
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

        if v2 < lim:
            print('velocidade maxima de agua menor que lim')
            import pdb; pdb.set_trace()

        return v2, h2, dfds2

    def run(self):
        print('loop')

        t_ = 0.0
        loop = 0

        """
        self.set_sat_in()

        #self.set_lamb()
        self.set_lamb_2()

        #self.set_global_problem()
        self.set_global_problem_vf()
        #self.set_global_problem_gr_vf()
        self.calculate_prolongation_op_het()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf)
        mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))
        #self.solve_linear_problem_numpy()
        qmax, fi = self.div_max_3(self.pf_tag)
        self.cfl(fi, qmax)

        #calculo da pressao multiescala
        Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf), self.trilOP, self.nf), self.nc, self.nc)
        Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf, self.b), self.nc)
        self.Pc = self.solve_linear_problem(Tc, Qc, self.nc)
        self.set_Pc()
        self.Pms = self.multimat_vector(self.trilOP, self.nf, self.Pc)
        mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))
        self.calculate_p_end()
        self.set_erro()"""

        self.mb.write_file('new_out_bif{0}.vtk'.format(loop))

        """
        loop = 1
        t_ = t_ + self.delta_t
        while t_ <= self.t and loop <= self.loops:

            self.calculate_sat()
            #self.set_lamb()
            self.set_lamb_2()
            #self.set_global_problem()
            self.set_global_problem_vf()
            self.calculate_prolongation_op_het()
            self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf)
            mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))
            #self.solve_linear_problem_numpy()
            qmax, fi = self.div_max_2(self.pf_tag)
            self.cfl(fi, qmax)


            Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf), self.trilOP, self.nf), self.nc, self.nc)
            Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf, self.b), self.nc)
            self.Pc = self.solve_linear_problem(Tc, Qc, self.nc)
            self.set_Pc()
            self.Pms = self.multimat_vector(self.trilOP, self.nf, self.Pc)
            mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))
            self.calculate_p_end()
            self.set_erro()

            mb.write_file('new_out_bif{0}.vtk'.format(loop))
            loop = loop+1
            t_ = t_ + self.delta_t"""

    def run_2(self):
        #0
        self.prod_w = []
        self.prod_o = []
        t_ = 0.0
        self.tempo = t_
        self.loop = 0
        self.set_sat_in()
        #self.set_lamb()
        self.set_lamb_2()
        self.set_global_problem_vf_2()

        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, len(self.all_fine_vols_ic))
        self.organize_Pf()
        del self.Pf
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf_all))
        del self.Pf_all
        # self.create_flux_vector_pf()


        self.calculate_restriction_op_2()
        self.calculate_prolongation_op_het()
        self.organize_op()
        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(
        self.trilOR, self.trans_fine, self.nf_ic), self.trilOP, self.nf_ic), self.nc, self.nc)
        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf_ic, self.b), self.nc)
        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        del self.Tc
        del self.Qc
        self.Pms = self.multimat_vector(self.trilOP, self.nf_ic, self.Pc)
        del self.Pc
        del self.trilOP
        self.organize_Pms()
        del self.Pms
        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms_all))
        del self.Pms_all
        self.test_conservation_coarse()
        self.Neuman_problem_6()
        self.create_flux_vector_pms()
        self.erro_2()



        #self.Neuman_problem_4_3()
        #self.erro()
        # qmax, fi = self.div_max_3(self.pf_tag)
        self.cfl()
        #print('qmax')
        #print(qmax)
        #print('delta_t')
        #print(self.delta_t)
        # vmax, h, dfds = self.vel_max(self.pf_tag)
        # self.cfl_2(vmax, h, dfds)
        print('delta_t: {0}'.format(self.delta_t))
        print('loop: {0}'.format(self.loop))
        print('\n')

        with open('prod_{0}.txt'.format(self.loop), 'w') as arq:
            arq.write('tempo:{0}\n'.format(self.tempo))
            arq.write('prod_o:{0}\n'.format(sum(self.prod_o)))
            arq.write('prod_w:{0}\n'.format(sum(self.prod_w)))


        self.mb.write_file('new_out_bif{0}.vtk'.format(self.loop))
        # arquivo = os.path.join(self.principal, 'new_out_bif{0}.vtk'.format(self.loop))
        # shutil.copy(arquivo, self.caminho1)
        # os.unlink(arquivo)
        self.loop = 1
        t_ = t_ + self.delta_t
        self.tempo = t_
        print('t')
        print(t_)

        while t_ <= self.t and self.loop < self.loops:
            #1
            self.prod_w = []
            self.prod_o = []
            self.calculate_sat_2()
            self.set_lamb_2()
            #self.set_lamb()
            self.set_global_problem_vf_2()

            self.Pf = self.solve_linear_problem(self.trans_fine, self.b, len(self.all_fine_vols_ic))
            self.organize_Pf()
            del self.Pf
            self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf_all))
            del self.Pf_all
            # self.create_flux_vector_pf()


            #self.calculate_restriction_op_2()
            self.calculate_prolongation_op_het()
            self.organize_op()
            self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(
            self.trilOR, self.trans_fine, self.nf_ic), self.trilOP, self.nf_ic), self.nc, self.nc)
            self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf_ic, self.b), self.nc)
            self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
            del self.Tc
            del self.Qc
            self.Pms = self.multimat_vector(self.trilOP, self.nf_ic, self.Pc)
            del self.Pc
            del self.trilOP
            self.organize_Pms()
            del self.Pms
            self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms_all))
            del self.Pms_all
            self.test_conservation_coarse()
            self.Neuman_problem_6()
            self.create_flux_vector_pms()
            self.erro_2()


            #self.Neuman_problem_4_3()
            #self.erro()

            #qmax, fi = self.div_max_3(self.pf_tag)
            self.cfl()
            #vmax, h, dfds = self.vel_max(self.pf_tag)
            #self.cfl_2(vmax, h, dfds)
            print('delta_t: {0}'.format(self.delta_t))
            print('loop: {0}'.format(self.loop))
            print('\n')
            self.mb.write_file('new_out_bif{0}.vtk'.format(self.loop))
            # arquivo = os.path.join(self.principal, 'new_out_bif{0}.vtk'.format(self.loop))
            # shutil.copy(arquivo, self.caminho1)
            # os.unlink(arquivo)

            with open('prod_{0}.txt'.format(self.loop), 'w') as arq:
                arq.write('tempo:{0}\n'.format(self.tempo))
                arq.write('prod_o:{0}\n'.format(sum(self.prod_o)))
                arq.write('prod_w:{0}\n'.format(sum(self.prod_w)))


            self.loop += 1
            t_ = t_ + self.delta_t
            self.tempo = t_


        shutil.copytree(self.caminho1, self.pasta)
