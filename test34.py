import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import time
import math
import os
import shutil
import random


class MsClassic_mono:
    def __init__(self, ind = False):
        """
        ind = True quando se quer excluir da conta os volumes com pressao prescrita
        """

        self.read_structured()
        self.comm = Epetra.PyComm()
        self.mb = core.Core()
        self.mb.load_file('out.h5m')
        self.root_set = self.mb.get_root_set()
        self.mesh_topo_util = topo_util.MeshTopoUtil(self.mb)
        self.create_tags(self.mb)
        self.all_fine_vols = self.mb.get_entities_by_dimension(self.root_set, 3)
        self.primals = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.primal_id_tag]),
            np.array([None]))
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
        gids = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols, flat = True)
        self.map_gids_in_elems = dict(zip(gids, self.all_fine_vols))
        elem0 = self.map_gids_in_elems[0]
        self.ro = self.mb.tag_get_data(self.rho_tag, elem0, flat=True)[0]
        self.mi = self.mb.tag_get_data(self.mi_tag, elem0, flat=True)[0]
        self.gama = self.mb.tag_get_data(self.gama_tag, elem0, flat=True)[0]
        self.atualizar = self.mb.tag_get_data(self.atualizar_tag, elem0, flat=True)[0]
        self.flag_grav = self.mb.tag_get_data(self.flag_gravidade_tag, elem0, flat=True)[0]
        if self.flag_grav == 0:
            self.get_wells()
        else:
            self.get_wells_gr()

        self.set_perm()
        # self.set_perm_2()
        # self.read_perms_and_phi_spe10()
        self.nf = len(self.all_fine_vols)
        self.nc = len(self.primals)
        self.map_all_fine_vols = dict(zip(self.all_fine_vols, gids))

        # self.set_volumes_in_interface()


        if ind == False:
            self.map_all_fine_vols = dict(zip(self.all_fine_vols, gids))
            self.calculate_restriction_op()
            # self.run()
            self.run_3()
            # self.run_4()

        else:

            self.neigh_wells_d = [] #volumes da malha fina vizinhos as pocos de pressao prescrita
            #self.elems_wells_d = [] #elementos com pressao prescrita
            for volume in self.wells:
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                if volume in self.wells_d:

                    adjs_volume = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                    for adj in adjs_volume:

                        if (adj not in self.wells_d) and (adj not in self.neigh_wells_d):
                            self.neigh_wells_d.append(adj)

            self.all_fine_vols_ic = set(self.all_fine_vols) - set(self.wells_d) #volumes da malha fina que sao icognitas
            self.map_vols_ic = dict(zip(list(self.all_fine_vols_ic), range(len(self.all_fine_vols_ic))))
            self.map_vols_ic_2 = dict(zip(range(len(self.all_fine_vols_ic)), list(self.all_fine_vols_ic)))
            self.nf_ic = len(self.all_fine_vols_ic)

            # self.run_2()

    funcoes = [
       lambda x: (x/np.linalg.norm(x))*(x/np.linalg.norm(x)), # unitario positivo na direcao de x
       lambda k1, k2: (2*k1*k2)/(k1+k2) # retorna a permeabilidade equivalente
    ]

    def add_gr(self):

        cont = 0
        for primal in self.primals:
            soma = 0
            soma2 = 0
            soma3 = 0
            temp_glob_adj = []
            temp_k = []

            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            gids_in_primal = self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True)
            volumes_in_interface = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id)
            gids_in_interface = self.mb.tag_get_data(self.global_id_tag, volumes_in_interface, flat=True)
            id_gids = list(gids_in_primal) + list(gids_in_interface)
            id_gids = dict(zip(id_gids, range(len(id_gids))))
            std_map = Epetra.Map(len(id_gids), 0, self.comm)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            b = Epetra.Vector(std_map)
            A_np = np.zeros((len(id_gids), len(id_gids)))
            b_np = np.zeros(len(id_gids))

            for volume in set(fine_elems_in_primal) | set(volumes_in_interface):
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                volume_centroid = self.mesh_topo_util.get_average_position([volume])
                if global_volume in self.wells_d:
                    index = self.wells_d.index(global_volume)
                    b[id_gids[global_volume]] = self.set_p[index] #+ (self.tz-volume_centroid[2])*self.gama
                    b_np[id_gids[global_volume]] = self.set_p[index] + (self.tz-volume_centroid[2])*self.gama
                    A.InsertGlobalValues(id_gids[global_volume], [1.0], [id_gids[global_volume]])
                    A_np[id_gids[global_volume], id_gids[global_volume]] = 1.0

                elif volume in volumes_in_interface:
                #if volume in volumes_in_interface:
                    for adj in adj_volumes:
                        global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                        fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, adj, flat=True)
                        primal_adj = self.mb.tag_get_data(
                            self.primal_id_tag, int(fin_prim), flat=True)[0]
                        if primal_adj == primal_id:
                            pms_adj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                            pms_volume = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                            b[id_gids[global_volume]] = pms_volume - pms_adj
                            b_np[id_gids[global_volume]] = pms_volume - pms_adj
                            temp_k.append(1.0)
                            temp_glob_adj.append(id_gids[global_volume])
                            temp_k.append(-1.0)
                            temp_glob_adj.append(id_gids[global_adj])
                            A.InsertGlobalValues(id_gids[global_volume], temp_k, temp_glob_adj)
                            A_np[id_gids[global_volume], temp_glob_adj] = temp_k[:]
                            temp_k = []
                            temp_glob_adj = []

                else:
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    for adj in adj_volumes:
                        global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                        adj_centroid = self.mesh_topo_util.get_average_position([adj])
                        direction = adj_centroid - volume_centroid
                        uni = self.unitary(direction)
                        altura = adj_centroid[2]
                        z = uni[2]
                        kvol = np.dot(np.dot(kvol,uni),uni)
                        kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        kadj = np.dot(np.dot(kadj,uni),uni)
                        keq = self.kequiv(kvol, kadj)
                        keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))

                        if z == 1.0:
                            keq2 = keq*self.gama
                            soma2 = soma2 - keq2
                            soma3 = soma3 + (keq2*(self.tz-altura))
                            #print(altura)

                        temp_glob_adj.append(id_gids[global_adj])
                        temp_k.append(-keq)
                        soma = soma + keq

                    soma2 = soma2*(self.tz-volume_centroid[2])
                    soma2 = -(soma2 + soma3)
                    temp_k.append(soma)
                    temp_glob_adj.append(id_gids[global_volume])
                    A.InsertGlobalValues(id_gids[global_volume], temp_k, temp_glob_adj)
                    A_np[id_gids[global_volume], temp_glob_adj] = temp_k
                    temp_k = []
                    temp_glob_adj = []

                    if global_volume in self.wells_n:
                        index = self.wells_n.index(global_volume)
                        tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                        if tipo_de_poco == 1:
                            b[id_gids[global_volume]] += self.set_q[index] + soma2
                            b_np[id_gids[global_volume]] += self.set_q[index] + soma2
                        else:
                            b[id_gids[global_volume]] += -self.set_q[index] + soma2
                            b_np[id_gids[global_volume]] += -self.set_q[index] + soma2
                    else:
                        b[id_gids[global_volume]] += soma2
                        b_np[id_gids[global_volume]] += soma2

                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

            A.FillComplete()
            x = self.solve_linear_problem(A, b, len(id_gids))
            x_np = np.linalg.solve(A_np, b_np)
            #id_gids_2 = dict(zip(len(gids_in_primal), gids_in_primal))

            for volume in fine_elems_in_primal:
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                pms_volume = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                # self.mb.tag_set_data(self.gravidade_tag, volume, x[id_gids[global_volume]] + pms_volume)
            cont += 1

    def add_gr_2(self):

        zeros = np.zeros(len(self.all_fine_vols))

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        #self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)
        self.vect_gr = Epetra.Vector(std_map)

        i = 0

        my_pairs = set()

        for collocation_point_set in self.sets:

            i += 1

            childs = self.mb.get_child_meshsets(collocation_point_set)
            collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]
            primal_elem = self.mb.tag_get_data(self.fine_to_primal_tag, collocation_point,
                                           flat=True)[0]
            primal_id = self.mb.tag_get_data(self.primal_id_tag, int(primal_elem), flat=True)[0]

            primal_id = self.ident_primal[primal_id]

            support_vals_gr_tag = self.mb.tag_get_handle(
                "TMP_SUPPORT_VALS_GR {0}".format(primal_id), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)

            self.mb.tag_set_data(support_vals_gr_tag, self.all_fine_vols, zeros)
            #self.mb.tag_set_data(support_vals_tag, collocation_point, 1.0)

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
                        self.calculate_local_problem_add_gr(
                            elems_edg, c_vertices, support_vals_gr_tag, self.h2)

                    self.calculate_local_problem_add_gr(
                        elems_fac, c_edges, support_vals_gr_tag, self.h2)

                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_add_gr(
                    elems_vol, c_faces, support_vals_gr_tag, self.h2)


                vals = self.mb.tag_get_data(support_vals_gr_tag, elems_vol, flat=True)
                gids = self.mb.tag_get_data(self.global_id_tag, elems_vol, flat=True)
                primal_elems = self.mb.tag_get_data(self.fine_to_primal_tag, elems_vol,
                                               flat=True)

                for val, gid in zip(vals, gids):
                    if (gid, primal_id) not in my_pairs:
                        if val == 0.0:
                            pass
                        else:
                            self.vect_gr[gid] = val

                        my_pairs.add((gid, primal_id))

        #self.trilOP.FillComplete()

    def calculate_local_problem_add_gr(self, elems, lesser_dim_meshsets, support_vals_tag, h2):
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
            soma2 = 0
            soma3 = 0
            k_elem = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
            centroid_elem = self.mesh_topo_util.get_average_position([elem])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(
                np.asarray([elem]), 2, 3, 0)
            values = []
            ids = []
            for adj in adj_volumes:
                if adj in id_map:
                    global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_elem
                    uni = self.unitary(direction)
                    altura = centroid_adj[2]
                    z = uni[2]
                    k_elem = np.dot(np.dot(k_elem,uni),uni)
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    keq = self.kequiv(k_elem, kadj)
                    keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
                    if z == 1.0:
                        keq2 = keq*self.gama
                        soma2 = soma2 - keq2
                        soma3 = soma3 + (keq2*(self.tz-altura))

                    ids.append(id_map[adj])
                    values.append(-keq)
                    k_elem = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])

            soma2 = soma2*(self.tz-centroid_elem[2])
            soma2 = -(soma2 + soma3)
            values.append(-sum(values))
            idx = id_map[elem]
            ids.append(idx)
            A.InsertGlobalValues(idx, values, ids)
            b[id_map[elem]] += soma2

        A.FillComplete()
        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        # AZ_last, AZ_summary, AZ_warnings
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-9)

        self.mb.tag_set_data(support_vals_tag, elems, np.asarray(x))

    def calculate_local_problem_het(self, elems, lesser_dim_meshsets, support_vals_tag, h2):
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
            values, ids = self.mount_lines_2(elem, id_map)
            A.InsertGlobalValues(id_map[elem], values, ids)

        A.FillComplete()

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        # AZ_last, AZ_summary, AZ_warnings
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-9)

        self.mb.tag_set_data(support_vals_tag, elems, np.asarray(x))

    def calculate_local_problem_het_corretion(self, elems, boundary):
        std_map = Epetra.Map(len(elems), 0, self.comm)
        linear_vals = np.arange(0, len(elems))
        id_map = dict(zip(elems, linear_vals))

        b = Epetra.Vector(std_map)
        x = Epetra.Vector(std_map)

        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
        # A_np = np.zeros((len(elems), len(elems)))
        # b_np = np.zeros(len(elems))

        boundary_dirichlet = set(elems) & set(self.wells_d)
        boundary_neuman = set(elems) & set(self.wells_n)
        boundary_collocation_points = (set(elems) & self.set_of_collocation_points_elems) - (boundary_dirichlet | boundary_neuman)
        if len(boundary_collocation_points) > 0:
            self.mb.tag_set_data(self.corretion_tag,
                                 boundary_collocation_points,
                                 np.repeat(0.0, len(boundary_collocation_points)))

        for elem in boundary_dirichlet | boundary_collocation_points:
             idx = id_map[elem]
             A.InsertGlobalValues(idx, [1], [idx])
             b[idx] = self.mb.tag_get_data(self.corretion_tag, elem, flat=True)[0]
             # A_np[idx, idx] = 1.0
             # b_np[idx] = self.mb.tag_get_data(self.corretion_tag, elem, flat=True)[0]

        elems_ic = set(elems) - (boundary_dirichlet | boundary_collocation_points)

        for elem in elems_ic:
            k_elem = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
            centroid_elem = self.mesh_topo_util.get_average_position([elem])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(
                np.asarray([elem]), 2, 3, 0)
            values = []
            ids = []
            for adj in adj_volumes:
                if adj not in elems:
                    continue
                k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                centroid_adj = self.mesh_topo_util.get_average_position([adj])
                direction = centroid_adj - centroid_elem
                uni = self.unitary(direction)
                k_elem = np.dot(np.dot(k_elem,uni),uni)
                k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                k_adj = np.dot(np.dot(k_adj,uni),uni)
                keq = self.kequiv(k_elem, k_adj)
                #keq = keq/(np.dot(h2, uni))
                keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
                values.append(-keq)
                ids.append(id_map[adj])
                k_elem = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
            #1
            values.append(-sum(values))
            idx = id_map[elem]
            ids.append(idx)
            A.InsertGlobalValues(idx, values, ids)
            # A_np[idx, ids] = values[:]
            if elem in boundary_neuman:
                index = self.wells_n.index(elem)
                if elem in self.wells_inj:
                    b[id_map[elem]] += self.wells_n[index]
                    # b_np[id_map[elem]] += self.wells_n[index]
                else:
                    b[id_map[elem]] -= self.wells_n[index]
                    # b_np[id_map[elem]] -= self.wells_n[index]

        A.FillComplete()


        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        # AZ_last, AZ_summary, AZ_warnings
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-9)
        # x_np = np.linalg.solve(A_np, b_np)

        self.mb.tag_set_data(self.corretion_tag, elems, np.asarray(x))
        self.mb.tag_set_data(self.corretion_tag, boundary_dirichlet, np.repeat(0.0, len(boundary_dirichlet)))

    def calculate_local_problem_het_corretion_2(self, elems, lesser_dim_meshsets, corretion_tag):
        #0
        n = len(elems)
        std_map = Epetra.Map(len(elems), 0, self.comm)
        linear_vals = np.arange(0, len(elems))
        id_map = dict(zip(elems, linear_vals))
        boundary_elms = set()
        b = Epetra.Vector(std_map)
        x = Epetra.Vector(std_map)
        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
        A_np = np.zeros((n, n))
        b_np = np.zeros(n)
        l1 = []
        l2 = []
        for ms in lesser_dim_meshsets:
            #1
            lesser_dim_elems = self.mb.get_entities_by_handle(ms)
            for elem in lesser_dim_elems:
                #2
                gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
                if elem in boundary_elms:
                    continue
                #2
                boundary_elms.add(elem)
                idx = id_map[elem]
                A.InsertGlobalValues(idx, [1.0], [idx])
                b[idx] = self.mb.tag_get_data(corretion_tag, elem, flat=True)[0]
                A_np[idx, idx] = 1.0
                b_np[idx] = self.mb.tag_get_data(corretion_tag, elem, flat=True)[0]
        #0
        for elem in (set(elems) ^ boundary_elms):
            #1
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            if elem in self.wells_d:
                #2
                index = self.wells_d.index(elem)
                A.InsertGlobalValues(id_map[elem], [1.0], [id_map[elem]])
                p = A.ExtractGlobalRowCopy(id_map[elem])
                b[id_map[elem]] = self.set_p[index]
                A_np[id_map[elem], id_map[elem]] = 1.0
                b_np[id_map[elem]] = self.set_p[index]
                # print(p[0])
                # print(p[1])
                # print('gid')
                # print(gid)
                # import pdb; pdb.set_trace()
                continue
            #1
            values, ids = self.mount_lines_2(elem, id_map)
            A.InsertGlobalValues(id_map[elem], values, ids)
            A_np[id_map[elem], ids] = values
            if elem in self.wells_n:
                index = self.wells_n.index(elem)
                if elem in self.wells_inj:
                    b[id_map[elem]] += self.set_q[index]
                    b_np[id_map[elem]] += self.set_q[index]
                else:
                    b[id_map[elem]] += -self.set_q[index]
                    b_np[id_map[elem]] += -self.set_q[index]
            else:
                pass
        #0
        # e = np.array(b)
        # print(e)
        # for i in range(len(elems)):
        #     p = A.ExtractGlobalRowCopy(i)
        #     print('aqui')
        #     print('i')
        #     print(i)
        #     print('trilinos')
        #     print(p[0])
        #     print(p[1])
        #     print('numpy')
        #     print(A_np[i])
        #     print('\n')
        # import pdb; pdb.set_trace()
        A.FillComplete()
        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        # AZ_last, AZ_summary, AZ_warnings
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-9)

        x2 = np.linalg.solve(A_np, b_np)
        x1 = np.array(x)
        dif = x2 - x1
        # print(sum(dif))
        # import pdb; pdb.set_trace()

        self.mb.tag_set_data(corretion_tag, elems, np.asarray(x))

    def calculate_local_problem_het_corretion_gr(self, elems, lesser_dim_meshsets, col):
        #0
        boundary_elms = set()
        boundary_elms.add(col)
        for ms in lesser_dim_meshsets:
            #1
            lesser_dim_elems = self.mb.get_entities_by_handle(ms)
            for elem in lesser_dim_elems:
                #2
                if elem in boundary_elms:
                    continue
                #2
                boundary_elms.add(elem)
        #0
        all_vols_ic = set(elems) - boundary_elms
        gids_ic = self.mb.tag_get_data(self.global_id_tag, all_vols_ic, flat=True)
        dim = len(all_vols_ic)
        std_map = Epetra.Map(dim, 0, self.comm)
        linear_vals = np.arange(0, dim)
        id_gids = dict(zip(gids_ic, linear_vals))
        gids_bound = self.mb.tag_get_data(self.global_id_tag, boundary_elms, flat=True)
        gids = self.mb.tag_get_data(self.global_id_tag, elems, flat=True)
        b = Epetra.Vector(std_map)
        x = Epetra.Vector(std_map)
        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
        A_np = np.zeros((dim, dim))
        b_np = np.zeros(dim)
        for volume in all_vols_ic:
            #1
            soma = 0
            soma2 = 0
            soma3 = 0
            temp_glob_adj = []
            temp_k = []
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            for adj in adj_volumes:
                #2
                if adj in boundary_elms:
                    continue
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                uni = self.unitary(direction)
                altura = adj_centroid[2]
                z = uni[2]
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
                if z == 1.0:
                    #3
                    keq2 = keq*self.gama
                    soma2 = soma2 - keq2
                    soma3 = soma3 + (keq2*(self.tz-altura))
                    #print(altura)
                #2
                temp_glob_adj.append(id_gids[global_adj])
                temp_k.append(-keq)
                soma = soma + keq
            #1
            soma2 = soma2*(self.tz-volume_centroid[2])
            soma2 = (soma2 + soma3)
            b[id_gids[global_volume]] += soma2
            b_np[id_gids[global_volume]] += soma2
            temp_k.append(soma)
            temp_glob_adj.append(id_gids[global_volume])
            A.InsertGlobalValues(id_gids[global_volume], temp_k, temp_glob_adj)
            A_np[id_gids[global_volume], temp_glob_adj] = temp_k
            # temp_k = []
            # temp_glob_adj = []
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

        A.FillComplete()

        # for elem in elems:
        #     i = id_map[elem]
        #     gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
        #     print('A')
        #     print(A[i])
        #     print('b')
        #     print(b[i])
        #     print('i')
        #     print(i)
        #     print('gid')
        #     print(gid)
        #     print('\n')


        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        # AZ_last, AZ_summary, AZ_warnings
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-9)
        x_np = np.linalg.solve(A_np, b_np)
        import pdb; pdb.set_trace()



        # self.mb.tag_set_data(self.corretion2_tag, elems, np.asarray(x))

    def calculate_local_problem_het_2(self, elems, lesser_dim_meshsets, support_vals_tag, h2):
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
            global_volume = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            if global_volume in self.wells_d:
                idx = id_map[elem]
                A.InsertGlobalValues(idx, [1], [idx])
                b[idx] = self.mb.tag_get_data(support_vals_tag, elem, flat=True)[0]
                continue
            k_elem = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
            centroid_elem = self.mesh_topo_util.get_average_position([elem])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(
                np.asarray([elem]), 2, 3, 0)
            values = []
            ids = []
            for adj in adj_volumes:
                if adj in id_map:
                    k_adj = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_elem
                    uni = self.unitary(direction)
                    k_elem = np.dot(np.dot(k_elem,uni),uni)
                    k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    k_adj = np.dot(np.dot(k_adj,uni),uni)
                    keq = self.kequiv(k_elem, k_adj)
                    keq = keq*(np.dot(self.A, uni)*self.ro)/(self.mi*np.dot(self.h, uni))
                    #keq = keq/(np.dot(h2, uni))
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

    def calculate_local_problem_het_3(self, elems, lesser_dim_meshsets, support_vals_tag, h2):
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
            global_volume = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, elem, flat=True)
            id_primal = self.mb.tag_get_data(
                self.primal_id_tag, int(fin_prim), flat=True)[0]
            if global_volume in self.wells_d:
                idx = id_map[elem]
                A.InsertGlobalValues(idx, [1], [idx])
                b[idx] = self.mb.tag_get_data(support_vals_tag, elem, flat=True)[0]
                continue
            k_elem = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
            centroid_elem = self.mesh_topo_util.get_average_position([elem])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(
                np.asarray([elem]), 2, 3, 0)
            values = []
            ids = []
            for adj in adj_volumes:
                if adj in id_map:
                    k_adj = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_elem
                    uni = self.unitary(direction)
                    k_elem = np.dot(np.dot(k_elem,uni),uni)
                    k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    k_adj = np.dot(np.dot(k_adj,uni),uni)
                    keq = self.kequiv(k_elem, k_adj)
                    keq = keq*(np.dot(self.A, uni)*self.ro)/(self.mi*np.dot(self.h, uni))
                    #keq = keq/(np.dot(h2, uni))
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

    def calculate_local_problem_het_4(self, elems, lesser_dim_meshsets, support_vals_tag, h2):
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
            global_volume = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            valor = self.mb.tag_get_data(support_vals_tag, elem, flat=True)[0]
            if valor == 1.0:
                idx = id_map[elem]
                A.InsertGlobalValues(idx, [1], [idx])
                b[idx] = valor
                continue
            k_elem = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
            centroid_elem = self.mesh_topo_util.get_average_position([elem])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(
                np.asarray([elem]), 2, 3, 0)
            values = []
            ids = []
            for adj in adj_volumes:
                if adj in id_map:
                    k_adj = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_elem
                    uni = self.unitary(direction)
                    k_elem = np.dot(np.dot(k_elem,uni),uni)
                    k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    k_adj = np.dot(np.dot(k_adj,uni),uni)
                    keq = self.kequiv(k_elem, k_adj)
                    keq = keq*(np.dot(self.A, uni)*self.ro)/(self.mi*np.dot(self.h, uni))
                    #keq = keq/(np.dot(h2, uni))
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

    def calculate_local_problem_het_5(self, elems, lesser_dim_meshsets, support_vals_tag, h2):
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
            centroid_elem = self.mesh_topo_util.get_average_position([elem])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(
                np.asarray([elem]), 2, 3, 0)
            values = []
            ids = []
            for adj in adj_volumes:
                if adj in id_map:
                    k_adj = self.mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_elem
                    uni = self.unitary(direction)
                    k_elem = np.dot(np.dot(k_elem,uni),uni)
                    k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    k_adj = np.dot(np.dot(k_adj,uni),uni)
                    keq = self.kequiv(k_elem, k_adj)
                    keq = keq/(np.dot(h2, uni))
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

        for volume in self.wells_d:
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            index = self.wells_d.index(volume)
            pms = self.set_p[index]
            self.Pms[global_volume] = pms

    def calculate_prolongation_op_het(self):

        zeros = np.zeros(len(self.all_fine_vols))

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        i = 0

        my_pairs = set()

        for collocation_point_set in self.sets:

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
                            elems_edg, c_vertices, support_vals_tag, self.h2)

                    self.calculate_local_problem_het(
                        elems_fac, c_edges, support_vals_tag, self.h2)

                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_het(
                    elems_vol, c_faces, support_vals_tag, self.h2)


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

        #self.trilOP.FillComplete()

    def calculate_prolongation_op_het_2(self):

        zeros = np.zeros(len(self.all_fine_vols))

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        i = 0

        my_pairs = set()

        for collocation_point_set in self.sets:

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

                set_of_dirichlet = set()

                for volume in (set(self.wells) & set(elems_vol)):
                    global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                    if global_volume in self.wells_d:
                        self.mb.tag_set_data(support_vals_tag, volume, 1.0)
                        set_of_dirichlet.add(volume)

                c_faces = self.mb.get_child_meshsets(vol)

                for face in c_faces:
                    elems_fac = self.mb.get_entities_by_handle(face)
                    c_edges = self.mb.get_child_meshsets(face)

                    for edge in c_edges:
                        elems_edg = self.mb.get_entities_by_handle(edge)
                        c_vertices = self.mb.get_child_meshsets(edge)
                        # a partir desse ponto op de prolongamento eh preenchido
                        self.calculate_local_problem_het_2(
                            elems_edg, c_vertices, support_vals_tag, self.h2)

                    self.calculate_local_problem_het_2(
                        elems_fac, c_edges, support_vals_tag, self.h2)

                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_het_2(
                    elems_vol, c_faces, support_vals_tag, self.h2)


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

        self.trilOP.FillComplete()

    def calculate_prolongation_op_het_3(self):

        zeros = np.zeros(len(self.all_fine_vols))

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        i = 0

        my_pairs = set()

        for collocation_point_set in self.sets:

            i += 1

            childs = self.mb.get_child_meshsets(collocation_point_set)
            collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]
            primal_elem = self.mb.tag_get_data(self.fine_to_primal_tag, collocation_point,
                                           flat=True)[0]
            _primal_id = self.mb.tag_get_data(self.primal_id_tag, int(primal_elem), flat=True)[0]

            primal_id = self.ident_primal[_primal_id]

            support_vals_tag = self.mb.tag_get_handle(
                "TMP_SUPPORT_VALS {0}".format(primal_id), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)

            self.mb.tag_set_data(support_vals_tag, self.all_fine_vols, zeros)
            self.mb.tag_set_data(support_vals_tag, collocation_point, 1.0)

            cont = 0
            for vol in childs:
                elems_vol = self.mb.get_entities_by_handle(vol)

                for volume in (set(self.wells) & set(elems_vol)):
                    global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                    fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, volume, flat=True)
                    primal_volume = self.mb.tag_get_data(
                        self.primal_id_tag, int(fin_prim), flat=True)[0]
                    if global_volume in self.wells_d and primal_volume == _primal_id:
                        self.mb.tag_set_data(support_vals_tag, volume, 1.0)

                c_faces = self.mb.get_child_meshsets(vol)

                for face in c_faces:
                    elems_fac = self.mb.get_entities_by_handle(face)
                    c_edges = self.mb.get_child_meshsets(face)

                    for edge in c_edges:
                        elems_edg = self.mb.get_entities_by_handle(edge)
                        c_vertices = self.mb.get_child_meshsets(edge)
                        # a partir desse ponto op de prolongamento eh preenchido
                        self.calculate_local_problem_het_3(
                            elems_edg, c_vertices, support_vals_tag, self.h2)

                    self.calculate_local_problem_het_3(
                        elems_fac, c_edges, support_vals_tag, self.h2)

                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_het_3(
                    elems_vol, c_faces, support_vals_tag, self.h2)


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

        self.trilOP.FillComplete()

    def calculate_prolongation_op_het_4(self):

        set_primals_wells_d = set([(0, 9), (9, 0), (9, 18), (18, 9), (8, 17), (17, 8), (17, 26), (26, 17)])

        zeros = np.zeros(len(self.all_fine_vols))

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        i = 0

        my_pairs = set()

        for collocation_point_set in self.sets:

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

                for volume in (set(self.wells) & set(elems_vol)):
                    global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                    fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, volume, flat=True)
                    primal_volume = self.mb.tag_get_data(
                        self.primal_id_tag, int(fin_prim), flat=True)[0]
                    primal_volume = self.ident_primal[primal_volume]
                    if (primal_id, primal_volume) in set_primals_wells_d or primal_id == primal_volume:
                        self.mb.tag_set_data(support_vals_tag, volume, 1.0)

                c_faces = self.mb.get_child_meshsets(vol)

                for face in c_faces:
                    elems_fac = self.mb.get_entities_by_handle(face)
                    c_edges = self.mb.get_child_meshsets(face)

                    for edge in c_edges:
                        elems_edg = self.mb.get_entities_by_handle(edge)
                        c_vertices = self.mb.get_child_meshsets(edge)
                        # a partir desse ponto op de prolongamento eh preenchido
                        self.calculate_local_problem_het_4(
                            elems_edg, c_vertices, support_vals_tag, self.h2)

                    self.calculate_local_problem_het_4(
                        elems_fac, c_edges, support_vals_tag, self.h2)

                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_het_4(
                    elems_vol, c_faces, support_vals_tag, self.h2)


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

        self.trilOP.FillComplete()

    def calculate_prolongation_op_het_5(self):

        zeros = np.zeros(len(self.all_fine_vols))

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        i = 0

        my_pairs = set()

        for collocation_point_set in self.sets:

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
                            elems_edg, c_vertices, support_vals_tag, self.h2)

                    self.calculate_local_problem_het(
                        elems_fac, c_edges, support_vals_tag, self.h2)

                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_het(
                    elems_vol, c_faces, support_vals_tag, self.h2)


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

        self.trilOP.FillComplete()

    def calculate_pwf(self, p_tag):
        lamb = 1.0

        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]

        vx = np.array([1, 0, 0])
        vy = np.array([0, 1, 0])

        for volume in self.wells:

            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            kx = np.dot(np.dot(kvol,vx),vx)
            ky = np.dot(np.dot(kvol,vy),vy)

            c0 = (kx*ky)**(1/2)
            c1 = ky/kx
            c2 = kx/ky
            c3 = ((c1)**(1/2))*(hx**2)
            c4 = ((c2)**(1/2))*(hy**2)
            c5 = c1**(1/4)
            c6 = c2**(1/4)
            req = 0.28*(((c3 + c4)**(1/2))/(c5 + c6))
            rw = self.mb.tag_get_data(self.raio_do_poco_tag, volume)[0][0]

            # WI = (2*math.pi*hz*c0)/(self.mi*(np.log(rw/req)))
            WI = (2*math.pi*hz*c0)/(self.mi*np.log(req/rw))

            tipo_de_prescricao = self.mb.tag_get_data(self.tipo_de_prescricao_tag, volume)[0][0]
            pvol = self.mb.tag_get_data(p_tag, volume)[0][0]

            if tipo_de_prescricao == 0:
                soma1 = 0.0
                soma2 = 0.0
                # pvol = self.mb.tag_get_data(p_tag, volume)[0][0]
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                volume_centroid = self.mesh_topo_util.get_average_position([volume])

                for adj in adjs_vol:
                    padj = self.mb.tag_get_data(p_tag, adj)[0][0]
                    adj_centroid = self.mesh_topo_util.get_average_position([adj])
                    direction = adj_centroid - volume_centroid
                    uni = self.unitary(direction)
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    keq = self.kequiv(kvol, kadj)
                    soma1 += (padj - pvol)*keq*(np.dot(self.A, uni))/(np.dot(self.h, uni)*self.mi)
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

                q = soma1
                pwf1 = pvol - (q/(WI))
                # pwf2 = pvol - (q/(WI2))

            else:
                q = self.mb.tag_get_data(self.valor_da_prescricao_tag, volume)[0][0]
                pwf1 = pvol - (q/(WI))
                # pwf2 = pvol - (q/(WI2))
            #print(pwf1)
            self.mb.tag_set_data(self.pwf_tag, volume, pwf1)

    def calculate_restriction_op(self):

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        self.trilOR = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        for primal in self.primals:

            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            #primal_id = self.ident_primal[primal_id]
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
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic), 0, self.comm)
        self.trilOR = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
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

    def corretion_func(self):
        #0
        zeros = np.zeros(len(self.all_fine_vols))
        self.mb.tag_set_data(self.corretion_tag, self.all_fine_vols, zeros)
        del zeros
        my_vols = set()
        # std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        i = 0
        for collocation_point_set in self.sets:
            #1
            i += 1
            childs = self.mb.get_child_meshsets(collocation_point_set)
            collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]
            primal_elem = self.mb.tag_get_data(self.fine_to_primal_tag, collocation_point,
                                           flat=True)[0]
            primal_id = self.mb.tag_get_data(self.primal_id_tag, int(primal_elem), flat=True)[0]
            # corretion_tag usa o mapeamento do primal_id
            primal_id = self.ident_primal[primal_id]

            # self.mb.tag_set_data(corretion_tag, self.all_fine_vols, zeros)
            for vol in childs:
                #2
                elems_vol = self.mb.get_entities_by_handle(vol)
                elems__vol = tuple(elems_vol)
                if elems__vol in my_vols:
                    continue
                my_vols.add(elems__vol)
                boundary = set(elems_vol) & set(self.wells)

                for volume in (boundary & set(self.wells_d)):
                    index = self.wells_d.index(volume)
                    p = self.set_p[index]
                    self.mb.tag_set_data(self.corretion_tag, volume, p)

                self.calculate_local_problem_het_corretion(
                    elems_vol, boundary)

    def corretion_func_2(self):

        zeros = np.zeros(len(self.all_fine_vols))

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        self.tril_corretion = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        i = 0

        my_pairs = set()

        for collocation_point_set in self.sets:

            i += 1

            childs = self.mb.get_child_meshsets(collocation_point_set)
            collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]
            primal_elem = self.mb.tag_get_data(self.fine_to_primal_tag, collocation_point,
                                           flat=True)[0]
            primal_id = self.mb.tag_get_data(self.primal_id_tag, int(primal_elem), flat=True)[0]

            primal_id = self.ident_primal[primal_id]

            corretion_tag = self.mb.tag_get_handle(
                "CORRETION {0}".format(primal_id), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)

            self.mb.tag_set_data(corretion_tag, self.all_fine_vols, zeros)

            # self.mb.tag_set_data(support_vals_tag, collocation_point, 1.0)

            for vol in childs:

                elems_vol = self.mb.get_entities_by_handle(vol)
                elems_dirichlet = set(self.wells_d) & set(elems_vol)
                for elem in elems_dirichlet:
                    index = self.wells_d.index(elem)
                    self.mb.tag_set_data(corretion_tag, elem, self.set_p[index])
                c_faces = self.mb.get_child_meshsets(vol)

                for face in c_faces:
                    elems_fac = self.mb.get_entities_by_handle(face)
                    c_edges = self.mb.get_child_meshsets(face)

                    for edge in c_edges:
                        elems_edg = self.mb.get_entities_by_handle(edge)
                        c_vertices = self.mb.get_child_meshsets(edge)
                        # a partir desse ponto op de prolongamento eh preenchido
                        self.calculate_local_problem_het_corretion_2(
                            elems_edg, c_vertices, corretion_tag)

                    self.calculate_local_problem_het_corretion_2(
                        elems_fac, c_edges, corretion_tag)

                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_het_corretion_2(
                    elems_vol, c_faces, corretion_tag)


                vals = self.mb.tag_get_data(corretion_tag, elems_vol, flat=True)
                gids = self.mb.tag_get_data(self.global_id_tag, elems_vol, flat=True)
                primal_elems = self.mb.tag_get_data(self.fine_to_primal_tag, elems_vol,
                                               flat=True)

                for val, gid in zip(vals, gids):
                    if (gid, primal_id) not in my_pairs:
                        if val == 0.0:
                            pass
                        else:
                            self.tril_corretion.InsertGlobalValues([gid], [primal_id], val)

                        my_pairs.add((gid, primal_id))

        #self.trilOP.FillComplete()

    def corretion_func_gr(self):
        #0

        zeros = np.zeros(len(self.all_fine_vols))
        self.mb.tag_set_data(self.corretion2_tag, self.all_fine_vols, zeros)
        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        i = 0
        my_vols = set()
        for collocation_point_set in self.sets:
            #1
            i += 1
            childs = self.mb.get_child_meshsets(collocation_point_set)
            collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]
            primal_elem = self.mb.tag_get_data(self.fine_to_primal_tag, collocation_point,
                                           flat=True)[0]
            primal_id = self.mb.tag_get_data(self.primal_id_tag, int(primal_elem), flat=True)[0]
            # corretion_tag usa o mapeamento do primal_id
            primal_id = self.ident_primal[primal_id]
            for vol in childs:
                #2
                elems_vol = self.mb.get_entities_by_handle(vol)
                if elems_vol in my_vols:
                    continue
                my_vols.add(elems_vol)
                gids = self.mb.tag_get_data(self.global_id_tag, elems_vol, flat=True)
                c_faces = self.mb.get_child_meshsets(vol)
                self.calculate_local_problem_het_corretion_gr(
                elems_vol, c_faces, collocation_point)
                # boundary = set(elems_vol) & set(self.elems_wells_d)
                # gids_bound = self.mb.tag_get_data(self.global_id_tag, boundary, flat=True)

    def create_flux_vector_pf(self):
        t1 = time.time()
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume
        """
        verif_local = 1
        lim4 = 1e-4
        soma = 0
        soma2 = 0
        self.store_flux_pf = {}

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
            self.store_flux_pf[volume] = flux
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
            arq.write('soma_prod:{0}'.format(sum(soma_prod)))

        print('soma_inj:{0}'.format(sum(soma_inj)))
        print('soma_prod:{0}'.format(sum(soma_prod)))

        print('soma2 : {0}'.format(soma2))
        if soma2 > lim4:
            print('nao esta dando conservativo globalmente')
            import pdb; pdb.set_trace()

        # print('saiu de def create_flux_vector_pf')
        print('\n')

        t2 = time.time()
        # import pdb; pdb.set_trace()

    def create_flux_vector_pms(self):
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume
        """
        soma_prod = 0
        soma_inj = 0
        lim4 = 1e-5
        self.store_velocity = {}
        self.store_flux = {}
        for primal in self.primals:
            #1
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_interface, volumes_in_primal = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id, flag = 1)
            for volume in fine_elems_in_primal:
                #2
                flux = {}
                velocity = {}
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                centroid_volume = self.mesh_topo_util.get_average_position([volume])
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                for adj in adjs_vol:
                    #3
                    gid_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
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
                    keq = self.kequiv(kvol, kadj)/(self.mi)
                    keq2 = keq
                    keq = keq*(np.dot(self.A, uni))
                    pvol2 = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    padj2 = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                    grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                    grad_p2 = (padj2 - pvol2)/float(abs(np.dot(direction, uni)))
                    q = (grad_p)*keq
                    if gid_adj > gid_vol:
                        v = -(grad_p2)*keq2
                    else:
                        v = (grad_p2)*keq2

                    flux[tuple(unit)] = q
                    velocity[tuple(unit)] = v
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

                #2
                # print(gid_vol)
                # print(velocity)
                # print('\n')
                # import pdb; pdb.set_trace()
                self.store_flux[volume] = flux
                self.mb.tag_set_data(self.flux_fine_pms_tag, volume, sum(flux.values()))
                # flt = sum(flux.values())
                # if volume not in self.wells_inj and volume not in self.wells_prod:
                #     lim4 = 1e-7
                #     if abs(flt) > lim4:
                #         print(gid_vol)
                #         print(flt)
                #         import pdb; pdb.set_trace()
                # flt = sum(flux.values())
                self.store_velocity[volume] = velocity

        for volume in set(self.all_fine_vols) - set(self.wells):
            gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            values = self.store_flux[volume].values()
            if sum(values) > lim4:
                print('fluxo multiescala nao esta dando conservativo')
                print('gid:{0}'.format(gid))
                print(sum(values))
                import pdb; pdb.set_trace()

        with open('fluxo_multiescala.txt', 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat= True)[0]
                values = self.store_flux[volume].values()
                if volume in self.wells_inj:
                    soma_inj += sum(values)
                else:
                    soma_prod += sum(values)
                arq.write('gid:{0} , fluxo:{1}\n'.format(gid, sum(values)))
            arq.write('\n')
            arq.write('soma_inj:{0}\n'.format(soma_inj))
            arq.write('soma_prod:{0}\n'.format(soma_prod))

    def create_tags(self, mb):

        # self.corretion_tag = self.mb.tag_get_handle(
        #     "CORRETION", 1, types.MB_TYPE_DOUBLE, True,
        #     types.MB_TAG_SPARSE, default_value=0.0)

        self.corretion2_tag = mb.tag_get_handle(
                        "CORRETION2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.flux_coarse_tag = mb.tag_get_handle(
                        "FLUX_COARSE", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.flux_fine_pms_tag = mb.tag_get_handle(
                        "FLUX_FINE_PMS", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.flux_fine_pf_tag = mb.tag_get_handle(
                        "FLUX_FINE_PF", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.Pc2_tag = mb.tag_get_handle(
                        "PC2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.fi_tag = mb.tag_get_handle(
                        "FI", 1, types.MB_TYPE_DOUBLE,
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

        self.pcorr_tag = mb.tag_get_handle(
                        "P_CORR", 1, types.MB_TYPE_DOUBLE,
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

        self.pw_tag = mb.tag_get_handle(
                        "PW", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.qw_tag = mb.tag_get_handle(
                        "QW", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pms_tag = mb.tag_get_handle(
                        "PMS", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pms2_tag = mb.tag_get_handle(
                        "PMS2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pms3_tag = mb.tag_get_handle(
                        "PMS3", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.p_tag = mb.tag_get_handle(
                        "P", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.qw_tag = mb.tag_get_handle(
                        "QW", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.perm_tag = mb.tag_get_handle(
                        "PERM", 9, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.volumes_in_primal_tag = mb.tag_get_handle(
            "VOLUMES_IN_PRIMAL", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_MESH, True)

        self.volumes_in_interface_tag = mb.tag_get_handle(
            "VOLUMES_IN_INTERFACE", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_MESH, True)

        self.global_id_tag = mb.tag_get_handle("GLOBAL_ID")

        self.collocation_point_tag = mb.tag_get_handle("COLLOCATION_POINT")

        self.elem_primal_id_tag = mb.tag_get_handle(
            "FINE_PRIMAL_ID", 1, types.MB_TYPE_INTEGER, True,
            types.MB_TAG_SPARSE)

        self.atualizar_tag = mb.tag_get_handle("ATUALIZAR")
        self.primal_id_tag = mb.tag_get_handle("PRIMAL_ID")
        self.fine_to_primal_tag = mb.tag_get_handle("FINE_TO_PRIMAL")
        self.valor_da_prescricao_tag = mb.tag_get_handle("VALOR_DA_PRESCRICAO")
        self.raio_do_poco_tag = mb.tag_get_handle("RAIO_DO_POCO")
        self.tipo_de_prescricao_tag = mb.tag_get_handle("TIPO_DE_PRESCRICAO")
        self.tipo_de_poco_tag = mb.tag_get_handle("TIPO_DE_POCO")
        self.tipo_de_fluido_tag = mb.tag_get_handle("TIPO_DE_FLUIDO")
        self.wells_tag = mb.tag_get_handle("WELLS")
        self.wells_n_tag = mb.tag_get_handle("WELLS_N")
        self.wells_d_tag = mb.tag_get_handle("WELLS_D")
        self.pwf_tag = mb.tag_get_handle("PWF")
        self.flag_gravidade_tag = mb.tag_get_handle("GRAV")
        self.gama_tag = mb.tag_get_handle("GAMA")
        self.rho_tag = mb.tag_get_handle("RHO")
        self.mi_tag = mb.tag_get_handle("MI")

    def erro(self):
        for volume in self.all_fine_vols:
            if volume in self.wells_d:
                erro = 0.0
                self.mb.tag_set_data(self.err_tag, volume, erro)
                self.mb.tag_set_data(self.err2_tag, volume, erro)
                continue

            Pf = self.mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            Pms = self.mb.tag_get_data(self.pms_tag, volume, flat = True)[0]
            erro = 100*(abs((Pf - Pms)/float(Pf)))
            self.mb.tag_set_data(self.err_tag, volume, erro)
            erro2 = abs(Pf-Pms)
            self.mb.tag_set_data(self.err2_tag, volume, erro2)

    def erro_2(self):
        for volume in self.all_fine_vols:
            if volume in self.wells_d:
                erro = 0.0
                self.mb.tag_set_data(self.err2_tag, volume, erro)
                continue
            Pf = self.mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            Pms = self.mb.tag_get_data(self.pms_tag, volume, flat = True)[0]
            erro = abs(Pf - Pms)#/float(abs(Pf))
            self.mb.tag_set_data(self.err2_tag, volume, erro)

    def erro_3(self):
        """
        pode ser substituído no lugar de self.erro_2
        """
        for volume in self.all_fine_vols:
            Pf = self.mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            Pms = self.mb.tag_get_data(self.pms_tag, volume, flat = True)[0]
            erro = 100*(abs(Pf/float(Pms) - 1.0))
            self.mb.tag_set_data(self.err_tag, volume, erro)

    def erro_4(self):
        for volume in self.all_fine_vols:
            Pf = self.mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            Pms = self.mb.tag_get_data(self.pms2_tag, volume, flat = True)[0]
            erro = abs(Pf - Pms)#/float(abs(Pf))
            self.mb.tag_set_data(self.err2_tag, volume, erro)

    def get_volumes_in_interfaces(self, fine_elems_in_primal, primal_id, **options):

        """
        obtem uma lista com os elementos dos primais adjacentes que estao na interface do primal corrente
        (primal_id)

        se flag == 1 alem dos volumes na interface dos primais adjacentes (volumes_in_interface)
        retorna tambem os volumes no primal que estao na interface (volumes_in_primal)

        se flag == 2 retorna apenas os volumes do primal corrente que estao na interface (volumes_in_primal)

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
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, adj, flat=True)
                primal_adj = self.mb.tag_get_data(self.primal_id_tag, int(fin_prim), flat=True)[0]
                if primal_adj != primal_id:
                    #3

                    # print('global volume: {0}'.format(global_volume))
                    # print('primal_id vol: {0}'.format(primal_id))
                    # print('global adj: {0}'.format(global_adj))
                    # print('primal_id adj: {0}'.format(primal_adj))
                    # print('\n')
                    # import pdb; pdb.set_trace()
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
        obtem:
        self.wells == os elementos que contem os pocos
        self.wells_d == lista contendo os ids globais dos volumes com pressao prescrita
        self.wells_n == lista contendo os ids globais dos volumes com vazao prescrita
        self.set_p == lista com os valores da pressao referente a self.wells_d
        self.set_q == lista com os valores da vazao referente a self.wells_n
        self.wells_inj == pocos injetores
        self.wells_prod = pocos produtores

        """
        wells_d = []
        wells_n = []
        set_p = []
        set_q = []
        wells_inj = []
        wells_prod = []

        wells_set = self.mb.tag_get_data(self.wells_tag, 0, flat=True)[0]
        self.wells = self.mb.get_entities_by_handle(wells_set)
        wells = self.wells

        for well in wells:
            global_id = self.mb.tag_get_data(self.global_id_tag, well, flat=True)[0]
            valor_da_prescricao = self.mb.tag_get_data(self.valor_da_prescricao_tag, well, flat=True)[0]
            tipo_de_prescricao = self.mb.tag_get_data(self.tipo_de_prescricao_tag, well, flat=True)[0]
            #raio_do_poco = self.mb.tag_get_data(self.raio_do_poco_tag, well, flat=True)[0]
            tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, well, flat=True)[0]
            #tipo_de_fluido = self.mb.tag_get_data(self.tipo_de_fluido_tag, well, flat=True)[0]
            #pwf = self.mb.tag_get_data(self.pwf_tag, well, flat=True)[0]
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

    def get_wells_gr(self):
        """
        obtem:
        self.wells == os elementos que contem os pocos
        self.wells_d == lista contendo os ids globais dos volumes com pressao prescrita
        self.wells_n == lista contendo os ids globais dos volumes com vazao prescrita
        self.set_p == lista com os valores da pressao referente a self.wells_d
        self.set_q == lista com os valores da vazao referente a self.wells_n

        """
        wells_d = []
        wells_n = []
        set_p = []
        set_q = []
        wells_inj = []
        wells_prod = []

        wells_set = self.mb.tag_get_data(self.wells_tag, 0, flat=True)[0]
        self.wells = self.mb.get_entities_by_handle(wells_set)
        wells = self.wells

        for well in wells:
            global_id = self.mb.tag_get_data(self.global_id_tag, well, flat=True)[0]
            valor_da_prescricao = self.mb.tag_get_data(self.valor_da_prescricao_tag, well, flat=True)[0]
            tipo_de_prescricao = self.mb.tag_get_data(self.tipo_de_prescricao_tag, well, flat=True)[0]
            centroid = self.mesh_topo_util.get_average_position([well])
            #raio_do_poco = self.mb.tag_get_data(self.raio_do_poco_tag, well, flat=True)[0]
            tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, well, flat=True)[0]
            #tipo_de_fluido = self.mb.tag_get_data(self.tipo_de_fluido_tag, well, flat=True)[0]
            #pwf = self.mb.tag_get_data(self.pwf_tag, well, flat=True)[0]
            if tipo_de_prescricao == 0:
                wells_d.append(well)
                set_p.append(valor_da_prescricao + (self.tz - centroid[2])*self.gama)
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

    def kequiv(self,k1,k2):
        """
        obbtem o k equivalente entre k1 e k2

        """
        #keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
        keq = (2*k1*k2)/(k1+k2)

        return keq

    def load_ops(self):
        lim = 10**(-7)
        mat = np.load('operadores.npz')
        OP = mat['OP']
        OR = mat['OR']
        sz_op = OP.shape[0]
        std_map = Epetra.Map(sz_op, 0, self.comm)
        self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)
        self.trilOR = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        for i in range(sz_op):
            index = list(np.nonzero(OP[i])[0])
            values = list(OP[i, index])
            self.trilOP.InsertGlobalValues(i, values, index)

        for i in range(self.nc):
            index = list(np.nonzero(OR[i])[0])
            values = list(OR[i, index])
            self.trilOR.InsertGlobalValues(i, values, index)

        self.trilOP.FillComplete()
        self.trilOR.FillComplete()

    def modificando_op(self):

        for volume in self.wells_d:
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, volume, flat=True)
            primal_volume = self.mb.tag_get_data(
                self.primal_id_tag, int(fin_prim), flat=True)[0]
            primal_volume = self.ident_primal[primal_volume]
            p = self.trilOP.ExtractGlobalRowCopy(global_volume)
            index = p[1]
            map_index = dict(zip(index, range(len(index))))
            values = p[0]
            for i in index:
                if i == primal_volume:
                    values[map_index[i]] = 1.0
                else:
                    values[map_index[i]] = 0.0

            self.trilOP.ReplaceGlobalValues(global_volume, values, index)

    def modificando_Tc_Qc(self):
        """
        imposicao de pressao na malha grossa
        """
        l1 = [0, 9, 18]
        l2 = [8, 17, 26]
        std_map = Epetra.Map(self.nc, 0, self.comm)
        Tc2 = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        for i in range(self.nc):
            #1
            p = self.Tc.ExtractGlobalRowCopy(i)
            index = p[1]
            values = p[0]
            map_index = dict(zip(index, range(len(index))))
            if i not in l1+l2:
                #2
                Tc2.InsertGlobalValues(i, values, index)
            #1
            else:
                #2
                Tc2.InsertGlobalValues(i, [1], [i])
                if i in l1:
                    self.Qc[i] = 3.0
                else:
                    self.Qc[i] = 2.0

        self.Tc = Tc2
        self.Tc.FillComplete()

    def modificar_matriz(self, A, rows, columns):
        """
        Modifica a matriz A para o tamanho (rows x columns)

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
        Modifica o tamanho do vetor para nc

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
        lim = 1e-7
        gid1 = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        volume_centroid = self.mesh_topo_util.get_average_position([volume])
        adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        soma = 0.0
        temp_ids = []
        temp_k = []
        for adj in adj_volumes:
            #1
            # adj_centroid = self.mb.tag_get_data(self.centroid_tag, adj, flat=True)
            gid2 = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
            adj_centroid = self.mesh_topo_util.get_average_position([adj])
            direction = adj_centroid - volume_centroid
            # uni = self.unitary(direction)
            uni = self.funcoes[0](direction)
            kvol = np.dot(np.dot(kvol,uni),uni)
            kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
            kadj = np.dot(np.dot(kadj,uni),uni)
            keq = self.kequiv(kvol, kadj)
            keq = keq*(np.dot(self.A, uni))/float(abs(self.mi*np.dot(direction, uni)))
            temp_ids.append(map_id[adj])
            temp_k.append(-keq)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        #0
        temp_k.append(-sum(temp_k))
        temp_ids.append(map_id[volume])

        return temp_k, temp_ids

    def mount_lines_2(self, volume, map_id):
        """
        monta as linhas da matriz
        retorna o valor temp_k e o mapeamento temp_id
        map_id = mapeamento dos elementos
        verifica se a adjacencia esta nos elementos do mapeamento
        """
        #0
        # volume_centroid = self.mb.tag_get_data(self.centroid_tag, volume, flat=True)
        volume_centroid = self.mesh_topo_util.get_average_position([volume])
        adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        soma = 0.0
        temp_ids = []
        temp_k = []
        for adj in adj_volumes:
            #1
            if adj in map_id.keys():
                #2
                # adj_centroid = self.mb.tag_get_data(self.centroid_tag, adj, flat=True)
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                # uni = self.unitary(direction)
                uni = self.funcoes[0](direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                # keq = self.funcoes[1](kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/(self.mi*abs(np.dot(direction, uni)))
                temp_ids.append(map_id[adj])
                temp_k.append(-keq)
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
        temp_k.append(-sum(temp_k))
        temp_ids.append(map_id[volume])
        #0
        return temp_k, temp_ids

    def multimat_vector(self, A, row, b):
        """
        Multiplica a matriz A de ordem row x row pelo vetor de tamanho row

        """
        #0
        std_map = Epetra.Map(row, 0, self.comm)
        c = Epetra.Vector(std_map)
        A.Multiply(False, b, c)

        return c

    def Neuman_problem_4(self):
        """
        Recalcula as presssoes em cada volume da malha primal (reconstrucao das pressoes) da seguinte maneira:
        primeiro verifica se os volumes estao nos pocos com pressao prescrita e sua pressao eh setada;
        depois verifica qual eh o volume que eh vertice da malha dual e seta a pressao multiescala do mesmo;
        depois calcula as pressoes no interior do volume primal com as condicoes prescritas acima
        e com vazao prescrita na interface dada pelo gradiente da pressao multiescala

        """

        colocation_points = self.mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, self.collocation_point_tag, np.array([None]))

        sets = []
        for col in colocation_points:
            #col = mb.get_entities_by_handle(col)[0]
            sets.append(self.mb.get_entities_by_handle(col)[0])
        sets = set(sets)

        for primal in self.primals:

            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_interface = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id)
            all_volumes = list(fine_elems_in_primal) + volumes_in_interface
            all_volumes_ic = self.all_fine_vols_ic & set(all_volumes)
            gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, all_volumes_ic, flat=True)
            map_volumes = dict(zip(gids_vols_ic, range(len(gids_vols_ic))))
            std_map = Epetra.Map(len(all_volumes_ic), 0, self.comm)
            volumes_in_primal = list(fine_elems_in_primal)
            volumes_in_primal.extend(volumes_in_interface)
            id_map = dict(zip(volumes_in_primal, range(len(volumes_in_primal))))
            std_map = Epetra.Map(len(volumes_in_primal), 0, self.comm)
            b = Epetra.Vector(std_map)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            dim = len(volumes_in_primal)
            b_np = np.zeros(dim)
            A_np = np.zeros((dim, dim))

            for volume in volumes_in_primal:
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                temp_id = []
                temp_k = []
                centroid_volume = self.mb.tag_get_data(self.centroid_tag, volume, flat=True)
                k_vol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                adj_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)

                if volume in self.wells:
                    if volume in self.wells_d:
                        index = self.wells_d.index(volume)
                        valor_da_prescricao = self.set_p[index]
                        temp_k.append(1.0)
                        temp_id.append(id_map[volume])
                        b[id_map[volume]] = valor_da_prescricao
                        b_np[id_map[volume]] = valor_da_prescricao

                    else:
                        soma = 0.0
                        for adj in adj_vol:
                            if adj in volumes_in_primal:
                                centroid_adj = self.mb.tag_get_data(self.centroid_tag, adj, flat=True)
                                direction = centroid_adj - centroid_volume
                                uni = self.unitary(direction)
                                k_vol = np.dot(np.dot(k_vol,uni),uni)
                                k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                                k_adj = np.dot(np.dot(k_adj,uni),uni)
                                #keq = self.kequiv(k_vol, k_adj)
                                keq = self.kequiv(k_vol, k_adj)
                                keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
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
                        centroid_adj = self.mb.tag_get_data(self.centroid_tag, adj, flat=True)
                        direction = centroid_adj - centroid_volume
                        uni = self.unitary(direction)
                        k_vol = np.dot(np.dot(k_vol,uni),uni)
                        k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        k_adj = np.dot(np.dot(k_adj,uni),uni)
                        keq = self.kequiv(k_vol, k_adj)
                        #keq = keq/(np.dot(self.h2, uni))
                        keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
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
                self.mb.tag_set_data(self.pcorr_tag, volume, x[i])
                self.mb.tag_set_data(self.pms2_tag, volume, x_np[i])
        #0

    def Neuman_problem_4_gr(self):
        """
        Recalcula as presssoes em cada volume da seguinte maneira:
        primeiro verifica se os volumes estao nos pocos com pressao prescrita e sua pressao eh setada;
        depois verifica qual eh o volume que eh vertice da malha dual e seta a pressao multiescala do mesmo;
        depois calcula as pressoes no interior do volume primal com as condicoes prescritas acima
        e com vazao prescrita na interface dada pelo gradiente da pressao multiescala

        com gravidade

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
                soma = 0.0
                soma2 = 0.0
                soma3 = 0.0
                temp_id = []
                temp_k = []
                volume_centroid = self.mb.tag_get_data(self.centroid_tag, volume, flat=True)
                adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                if volume in sets:
                    #3
                    temp_k.append(1.0)
                    temp_id.append(map_volumes[global_volume])
                    b[map_volumes[global_volume]] = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    b_np[map_volumes[global_volume]] = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
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
                        adj_centroid = self.mb.tag_get_data(self.centroid_tag, volume, flat=True)
                        direction = adj_centroid - volume_centroid
                        altura = adj_centroid[2]
                        uni = self.unitary(direction)
                        z = uni[2]
                        kvol = np.dot(np.dot(kvol,uni),uni)
                        kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        kadj = np.dot(np.dot(kadj,uni),uni)
                        keq = self.kequiv(kvol, kadj)
                        keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
                        if z == 1.0:
                            #5
                            keq2 = keq*self.gama
                            soma2 = soma2 - keq2
                            soma3 = soma3 + (keq2*(self.tz-altura))
                        #4
                        else:
                            #5
                            pass
                        #4
                        if adj in self.wells_d:
                            #5
                            soma = soma + keq
                            index = self.wells_d.index(adj)
                            b[map_volumes[global_volume]] += self.set_p[index]*(keq)
                            b_np[map_volumes[global_volume]] += self.set_p[index]*(keq)
                        #4
                        else:
                            #5
                            temp_id.append(map_volumes[global_adj])
                            temp_k.append(-keq)
                            soma = soma + keq
                        #4
                        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    #3
                    soma2 = soma2*(self.tz-volume_centroid[2])
                    soma2 = -(soma2 + soma3)
                    temp_k.append(soma)
                    temp_id.append(map_volumes[global_volume])
                    if volume in self.wells_n:
                        #4
                        index = self.wells_n.index(volume)
                        tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                        if tipo_de_poco == 1:
                            #5
                            b[map_volumes[global_volume]] += self.set_q[index] + soma2
                            b_np[map_volumes[global_volume]] += self.set_q[index] + soma2
                        #4
                        else:
                            #5
                            b[map_volumes[global_volume]] += -self.set_q[index] + soma2
                            b_np[map_volumes[global_volume]] += -self.set_q[index] + soma2
                    #3
                    else:
                        #4
                        b[map_volumes[global_volume]] +=  soma2
                        b_np[map_volumes[global_volume]] += soma2
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
                index = self.wells_d.index(volume)
                p = self.set_p[index]
                self.mb.tag_set_data(self.pcorr_tag, volume, p)
                self.mb.tag_set_data(self.pms2_tag, volume, p)

    def Neuman_problem_4_gr_2(self):
        """
        Recalcula as presssoes em cada volume da seguinte maneira:
        primeiro verifica se os volumes estao nos pocos com pressao prescrita e sua pressao eh setada;
        depois verifica qual eh o volume que eh vertice da malha dual e seta a pressao multiescala do mesmo;
        depois calcula as pressoes no interior do volume primal com as condicoes prescritas acima
        e com vazao prescrita na interface dada pelo gradiente da pressao multiescala

        com gravidade

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
            map_volumes = dict(zip(all_volumes, range(len(all_volumes))))
            std_map = Epetra.Map(len(all_volumes), 0, self.comm)
            b = Epetra.Vector(std_map)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            dim = len(all_volumes)
            b_np = np.zeros(dim)
            A_np = np.zeros((dim, dim))
            for volume in all_volumes:
                #2
                soma = 0.0
                soma2 = 0.0
                soma3 = 0.0
                temp_id = []
                temp_k = []
                volume_centroid = self.mb.tag_get_data(self.centroid_tag, volume, flat=True)
                adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                if volume in self.wells_d:
                    #3
                    index = self.wells_d.index(volume)
                    temp_k.append(1.0)
                    temp_id.append(map_volumes[volume])
                    b[map_volumes[volume]] = self.set_p[index]
                    b_np[map_volumes[volume]] = self.set_p[index]
                #2
                elif volume in self.wells_n:
                    #3
                    for adj in adj_volumes:
                        #4
                        if adj in all_volumes:
                            #5
                            adj_centroid = self.mb.tag_get_data(self.centroid_tag, volume, flat=True)
                            direction = adj_centroid - volume_centroid
                            altura = adj_centroid[2]
                            uni = self.unitary(direction)
                            z = uni[2]
                            kvol = np.dot(np.dot(kvol,uni),uni)
                            kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                            kadj = np.dot(np.dot(kadj,uni),uni)
                            keq = self.kequiv(kvol, kadj)
                            keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
                            if z == 1.0:
                                #6
                                keq2 = keq*self.gama
                                soma2 = soma2 - keq2
                                soma3 = soma3 + (keq2*(self.tz-altura))
                            #5
                            temp_id.append(map_volumes[global_adj])
                            temp_k.append(-keq)
                            soma = soma + keq
                            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    #3
                    soma2 = soma2*(self.tz-volume_centroid[2])
                    soma2 = -(soma2 + soma3)
                    temp_k.append(soma)
                    temp_id.append(map_volumes[volume])
                    index = self.wells_n.index(volume)
                    if volume in self.wells_inj:
                        #4
                        b[map_volumes[volume]] += self.set_q[index] + soma2
                        b_np[map_volumes[volume]] += self.set_q[index] + soma2
                    #3
                    else:
                        #4
                        b[map_volumes[volume]] += -self.set_q[index] + soma2
                        b_np[map_volumes[volume]] += -self.set_q[index] + soma2
                #2
                elif volume in sets:
                    #3
                    temp_k.append(1.0)
                    temp_id.append(map_volumes[volume])
                    b[map_volumes[volume]] = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    b_np[map_volumes[volume]] = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                #2
                elif volume in volumes_in_interface:
                    #3
                    for adj in adj_volumes:
                        #4
                        if adj in fine_elems_in_primal:
                            #5
                            pms_adj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                            pms_volume = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                            b[map_volumes[volume]] = pms_volume - pms_adj
                            b_np[map_volumes[volume]] = pms_volume - pms_adj
                            temp_k.append(1.0)
                            temp_id.append(map_volumes[volume])
                            temp_k.append(-1.0)
                            temp_id.append(map_volumes[adj])
                #2
                else:
                    #3
                    for adj in adj_volumes:
                        #4
                        adj_centroid = self.mb.tag_get_data(self.centroid_tag, volume, flat=True)
                        direction = adj_centroid - volume_centroid
                        altura = adj_centroid[2]
                        uni = self.unitary(direction)
                        z = uni[2]
                        kvol = np.dot(np.dot(kvol,uni),uni)
                        kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        kadj = np.dot(np.dot(kadj,uni),uni)
                        keq = self.kequiv(kvol, kadj)
                        keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
                        if z == 1.0:
                            #5
                            keq2 = keq*self.gama
                            soma2 = soma2 - keq2
                            soma3 = soma3 + (keq2*(self.tz-altura))
                        #4
                        temp_id.append(map_volumes[adj])
                        temp_k.append(-keq)
                        soma = soma + keq
                        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    #3
                    soma2 = soma2*(self.tz-volume_centroid[2])
                    soma2 = -(soma2 + soma3)
                    temp_k.append(soma)
                    temp_id.append(map_volumes[volume])
                    b[map_volumes[volume]] +=  soma2
                    b_np[map_volumes[volume]] += soma2
                #2
                A.InsertGlobalValues(map_volumes[volume], temp_k, temp_id)
                A_np[map_volumes[volume], temp_id] = temp_k
            #1
            A.FillComplete()
            x = self.solve_linear_problem(A, b, dim)
            x_np = np.linalg.solve(A_np, b_np)
            for volume in all_volumes:
                #2
                self.mb.tag_set_data(self.pcorr_tag, volume, x[map_volumes[volume]])
                self.mb.tag_set_data(self.pms2_tag, volume, x_np[map_volumes[volume]])

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
                # y = map_volumes[volume]
                gid1 = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                centroid_volume = self.mesh_topo_util.get_average_position([volume])
                k_vol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                # print('in wells d: {0}'.format(volume in self.wells_d))
                # print('in collocation_points: {0}'.format(volume in self.set_of_collocation_points_elems))
                # print('in volumes_in_primal: {0}'.format(volume in volumes_in_primal))
                if volume in self.wells_d or volume in self.set_of_collocation_points_elems:
                    #3
                    value = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    temp_k.append(1.0)
                    temp_id.append(map_volumes[volume])
                    b[map_volumes[volume]] = value
                    # b_np[map_volumes[volume]] = value
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
                        k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        k_adj = np.dot(np.dot(k_adj,uni),uni)
                        keq = self.kequiv(k_vol, k_adj)
                        keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
                        if adj in all_volumes:
                            #5
                            soma += keq
                            temp_k.append(-keq)
                            temp_id.append(map_volumes[adj])
                        #4
                        else:
                            #5
                            q_in = (padj - pvol)*(keq)
                            b[map_volumes[volume]] += q_in
                            # b_np[map_volumes[volume]] += q_in

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
                            # b_np[map_volumes[volume]] += self.set_q[index]
                        #4
                        else:
                            #5
                            b[map_volumes[volume]] -= self.set_q[index]
                            # b_np[map_volumes[volume]] -= self.set_q[index]
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
                            # b_np[map_volumes[volume]] += self.set_q[index]
                        #4
                        else:
                            #5
                            b[map_volumes[volume]] -= self.set_q[index]
                            # b_np[map_volumes[volume]] -= self.set_q[index]
                #2
                A.InsertGlobalValues(map_volumes[volume], temp_k, temp_id)
                # A_np[map_volumes[volume], temp_id] = temp_k
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
            # x_np = np.linalg.solve(A_np, b_np)
            # print(x_np)
            for volume in all_volumes:
                #2
                self.mb.tag_set_data(self.pcorr_tag, volume, x[map_volumes[volume]])
                # self.mb.tag_set_data(self.pms3_tag, volume, x_np[map_volumes[volume]])

    def organize_op(self):
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
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

    def organize_or(self):
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        trilOR2 = []

        gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols_ic, flat=True)

    def organize_Pf(self):

        """
        organiza a solucao da malha fina para setar no arquivo de saida
        """
        #0
        # Pf = np.zeros(len(self.all_fine_vols))
        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)
        Pf2 = Epetra.Vector(std_map)
        for i in range(len(self.Pf)):
            #1
            value = self.Pf[i]
            elem = self.map_vols_ic_2[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pf2[gid] = value
            # Pf[ind] = value
        #0
        for i in range(len(self.wells_d)):
            #1
            value = self.set_p[i]
            elem = self.wells_d[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pf2[gid] = value
            # Pf[ind] = value
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

    def pymultimat(self, A, B, nf):
        """
        Multiplica a matriz A pela matriz B

        """

        nf_map = Epetra.Map(nf, 0, self.comm)

        C = Epetra.CrsMatrix(Epetra.Copy, nf_map, 3)

        EpetraExt.Multiply(A, False, B, False, C)

        C.FillComplete()

        return C

    def read_perms_and_phi_spe10(self):
        nx = 60
        ny = 220
        nz = 85
        N = nx*ny*nz
        # l1 = [N, 2*N, 3*N]
        # l2 = [0, 1, 2]
        #
        #
        # ks = np.loadtxt('spe_perm.dat')
        # t1, t2 = ks.shape
        # ks = ks.reshape((t1*t2))
        # ks2 = np.zeros((N, 9))
        #
        #
        # for i in range(0, N):
        #     # as unidades do spe_10 estao em milidarcy
        #     # unidade de darcy em metro quadrado =  (1 Darcy)*(9.869233e-13)
        #     # fonte -- http://www.calculator.org/property.aspx?name=permeability
        #     ks2[i, 0] = ks[i]*(10**(-3))# *9.869233e-13
        #
        # cont = 0
        # for i in range(N+1, 2*N):
        #     ks2[cont, 4] = ks[i]*(10**(-3))# *9.869233e-13
        #     cont += 1
        #
        # cont = 0
        # for i in range(2*N+1, 3*N):
        #     ks2[cont, 8] = ks[i]*(10**(-3))# *9.869233e-13
        #     cont += 1
        #
        # cont = None
        # phi = np.loadtxt('spe_phi.dat')
        # np.savez_compressed('spe10_perms_and_phi', perms = ks2, phi = phi)
        # ks2 = None

        # obter a permeabilidade de uma regiao
        # digitar o inicio e o fim da regiao

        ks = np.load('spe10_perms_and_phi.npz')['perms']
        phi = np.load('spe10_perms_and_phi.npz')['phi']

        # gid1 = [0, 0, 50]
        gid1 = [0, 0, 0]
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

        cont = 0

        for volume in self.all_fine_vols:
            self.mb.tag_set_data(self.perm_tag, volume, permeabilidade[cont])
            self.mb.tag_set_data(self.fi_tag, volume, fi[cont])
            cont += 1


        # self.mb.tag_set_data(self.perm_tag, self.all_fine_vols, permeabilidade)
        # self.mb.tag_set_data(self.fi_tag, self.all_fine_vols, fi)
        # for volume in self.all_fine_vols:
        #     gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)
        #     perm = self.mb.tag_get_data(self.perm_tag, volume).reshape([3,3])
        #     fi2 = self.mb.tag_get_data(self.fi_tag, volume, flat = True)[0]

    def read_structured(self):
        """
        Le os dados do arquivo structured

        """
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

        tx = nx*hx
        ty = ny*hy
        tz = nz*hz
        h = np.array([hx, hy, hz])
        h2 = np.array([hx**2, hy**2, hz**2])

        ax = hy*hz
        ay = hx*hz
        az = hx*hy
        A = np.array([ax, ay, az])

        V = hx*hy*hz

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.h2 = h2
        self.tz = tz
        self.h = h
        self.hm = h/2.0
        self.A = A
        self.V = V
        self.tam = np.array([tx, ty, tz])

    def save_ops(self):

        OP = np.zeros((self.nf_ic, self.nf_ic))
        OR = OP.copy()

        for i in range(self.nf_ic):
            p = self.trilOP.ExtractGlobalRowCopy(i)
            OP[i, p[1]] = p[0]

        for i in range(self.nc):
            p = self.trilOR.ExtractGlobalRowCopy(i)
            OR[i, p[1]] = p[0]

        np.savez_compressed('operadores', OP = OP, OR = OR)

    def set_contorno(self):
        self.mb.tag_set_data(self.contorno_tag, self.wells_inj, np.repeat(1.0, len(self.wells_inj)))
        self.mb.tag_set_data(self.contorno_tag, self.wells_prod, np.repeat(0.0, len(self.wells_prod)))
        elems = set(self.all_fine_vols) - set(self.wells)
        self.mb.tag_set_data(self.contorno_tag, elems, np.repeat(0.5, len(elems)))

    def set_global_problem(self):
        """
        Obtem a matriz de transmissibilidade por diferencas finitas

        """

        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)

        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)

        for volume in self.all_fine_vols:

            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]

            if global_volume not in self.wells_d:

                soma = 0.0
                temp_glob_adj = []
                temp_k = []

                for adj in adj_volumes:
                    global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    adj_centroid = self.mesh_topo_util.get_average_position([adj])

                    direction = adj_centroid - volume_centroid
                    uni = self.unitary(direction)

                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    keq = self.kequiv(kvol, kadj)
                    keq = keq/(np.dot(self.h2, uni))

                    temp_glob_adj.append(global_adj)
                    temp_k.append(-keq)

                    soma = soma + keq

                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

                #soma = -1*soma
                temp_k.append(soma)
                temp_glob_adj.append(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, temp_k, temp_glob_adj)

                if global_volume in self.wells_n:
                    index = self.wells_n.index(global_volume)
                    tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                    if tipo_de_poco == 1:
                        self.b[global_volume] = self.set_q[index]
                    else:
                        self.b[global_volume] = -self.set_q[index]

            else:
                index = self.wells_d.index(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, [1.0], [global_volume])
                self.b[global_volume] = self.set_p[index]

        self.trans_fine.FillComplete()

    def set_global_problem_gr(self):

        """
        transmissibilidade da malha fina com gravidade
        """

        self.ro = 1.0
        self.mi = 1.0
        self.gama = 1.0

        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)

        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)

        for volume in self.all_fine_vols:

            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]

            if global_volume not in self.wells_d:

                soma = 0.0
                soma2 = 0.0
                soma3 = 0.0
                temp_glob_adj = []
                temp_k = []

                for adj in adj_volumes:
                    global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    adj_centroid = self.mesh_topo_util.get_average_position([adj])
                    direction = adj_centroid - volume_centroid
                    altura = adj_centroid[2]
                    uni = self.unitary(direction)
                    z = uni[2]
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    keq = self.kequiv(kvol, kadj)
                    keq = keq/(np.dot(self.h2, uni))

                    if z == 1.0:
                        keq2 = keq*self.gama
                        soma2 = soma2 - keq2
                        soma3 = soma3 + (keq2*(self.tz-altura))

                    temp_glob_adj.append(global_adj)
                    temp_k.append(-keq)

                    soma = soma + keq

                soma2 = soma2*(self.tz-volume_centroid[2])
                soma2 = -(soma2 + soma3)
                #soma = -1*soma
                temp_k.append(soma)
                temp_glob_adj.append(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, temp_k, temp_glob_adj)

                if global_volume in self.wells_n:
                    index = self.wells_n.index(global_volume)
                    tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                    if tipo_de_poco == 1:
                        self.b[global_volume] = self.set_q[index] + soma2
                    else:
                        self.b[global_volume] = -self.set_q[index] + soma2
                else:
                    self.b[global_volume] = soma2

                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

            else:
                index = self.wells_d.index(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, [1.0], [global_volume])
                self.b[global_volume] = self.set_p[index]

        self.trans_fine.FillComplete()

    def set_global_problem_gr_vf(self):

        """
        transmissibilidade da malha fina com gravidade _volumes finitos
        """

        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)

        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)

        for volume in self.all_fine_vols:

            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]

            if global_volume not in self.wells_d:

                soma = 0.0
                soma2 = 0.0
                soma3 = 0.0
                temp_glob_adj = []
                temp_k = []

                for adj in adj_volumes:
                    global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    adj_centroid = self.mesh_topo_util.get_average_position([adj])
                    direction = adj_centroid - volume_centroid
                    altura = adj_centroid[2]
                    uni = self.unitary(direction)
                    z = uni[2]
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    keq = self.kequiv(kvol, kadj)
                    keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))

                    if z == 1.0:
                        keq2 = keq*self.gama
                        soma2 = soma2 - keq2
                        soma3 = soma3 + (keq2*(self.tz-altura))

                    temp_glob_adj.append(global_adj)
                    temp_k.append(-keq)

                    soma = soma + keq

                soma2 = soma2*(self.tz-volume_centroid[2])
                soma2 = -(soma2 + soma3)
                #soma = -1*soma
                temp_k.append(soma)
                temp_glob_adj.append(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, temp_k, temp_glob_adj)

                if global_volume in self.wells_n:
                    index = self.wells_n.index(global_volume)
                    tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                    if tipo_de_poco == 1:
                        self.b[global_volume] = self.set_q[index] + soma2
                    else:
                        self.b[global_volume] = -self.set_q[index] + soma2
                else:
                    self.b[global_volume] = soma2

                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

            else:
                index = self.wells_d.index(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, [1.0], [global_volume])
                self.b[global_volume] = self.set_p[index]

        self.trans_fine.FillComplete()

    def set_global_problem_gr_vf_2(self):

        """
        transmissibilidade da malha fina com gravidade _volumes finitos
        excluindo volumes com pressao prescrita
        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)
        for volume in self.all_fine_vols_ic - set(self.neigh_wells_d):
            #1
            soma = 0.0
            soma2 = 0.0
            soma3 = 0.0
            temp_glob_adj = []
            temp_k = []
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            for adj in adj_volumes:
                #2
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                altura = adj_centroid[2]
                uni = self.unitary(direction)
                z = uni[2]
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
                if z == 1.0:
                    #3
                    keq2 = keq*self.gama
                    soma2 = soma2 - keq2
                    soma3 = soma3 + (keq2*(self.tz-altura))
                #2
                temp_glob_adj.append(self.map_vols_ic[global_adj])
                temp_k.append(-keq)
                soma = soma + keq
            #1
            soma2 = soma2*(self.tz-volume_centroid[2])
            soma2 = -(soma2 + soma3)
            temp_k.append(soma)
            temp_glob_adj.append(self.map_vols_ic[global_volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[global_volume], temp_k, temp_glob_adj)
            if global_volume in self.wells_n:
                #2
                index = self.wells_n.index(global_volume)
                tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                if tipo_de_poco == 1:
                    #3
                    self.b[self.map_vols_ic[global_volume]] = self.set_q[index] + soma2
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[global_volume]] = -self.set_q[index] + soma2
            #1
            else:
                #2
                self.b[self.map_vols_ic[global_volume]] = soma2
            #1
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        #0
        for volume in self.neigh_wells_d:
            #1
            soma = 0.0
            soma2 = 0.0
            soma3 = 0.0
            temp_glob_adj = []
            temp_k = []
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            for adj in adj_volumes:
                #2
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                altura = adj_centroid[2]
                uni = self.unitary(direction)
                z = uni[2]
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/(self.mi*np.dot(self.h, uni))
                if z == 1.0:
                    #3
                    keq2 = keq*self.gama
                    soma2 = soma2 - keq2
                    soma3 = soma3 + (keq2*(self.tz-altura))
                #2
                if global_adj in self.wells_d:
                    #3
                    soma = soma + keq
                    index = self.wells_d.index(global_adj)
                    self.b[self.map_vols_ic[global_volume]] += self.set_p[index]*(keq)
                #2
                else:
                    #3
                    temp_glob_adj.append(self.map_vols_ic[global_adj])
                    temp_k.append(-keq)
                    soma = soma + keq
                #2
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            soma2 = soma2*(self.tz-volume_centroid[2])
            soma2 = -(soma2 + soma3)
            temp_k.append(soma)
            temp_glob_adj.append(self.map_vols_ic[global_volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[global_volume], temp_k, temp_glob_adj)
            if global_volume in self.wells_n:
                #2
                index = self.wells_n.index(global_volume)
                tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                if tipo_de_poco == 1:
                    #3
                    self.b[self.map_vols_ic[global_volume]] += self.set_q[index] + soma2
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[global_volume]] += -self.set_q[index] + soma2
            #1
            else:
                #2
                self.b[self.map_vols_ic[global_volume]] += soma2
        #0
        self.trans_fine.FillComplete()

    def set_global_problem_gr_vf_3(self):

        """
        transmissibilidade da malha fina com gravidade _volumes finitos
        excluindo volumes com pressao prescrita
        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)
        for volume in self.all_fine_vols_ic - set(self.neigh_wells_d):
            #1
            soma2 = 0.0
            soma3 = 0.0
            temp_glob_adj = []
            temp_k = []
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            for adj in adj_volumes:
                #2
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
                    soma2 = soma2 - keq2
                    soma3 = soma3 + (keq2*(self.tz-altura))
                #2
                temp_glob_adj.append(self.map_vols_ic[adj])
                temp_k.append(-keq)
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            soma2 = soma2*(self.tz-volume_centroid[2])
            soma2 = -(soma2 + soma3)
            temp_k.append(-sum(temp_k))
            temp_glob_adj.append(self.map_vols_ic[volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                index = self.wells_n.index(volume)
                tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                if volume in  self.wells_inj:
                    #3
                    self.b[self.map_vols_ic[volume]] += self.set_q[index] + soma2
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[volume]] += -self.set_q[index] + soma2
            #1
            else:
                #2
                self.b[self.map_vols_ic[volume]] += soma2
            #1
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        #0
        for volume in self.neigh_wells_d:
            #1
            soma = 0.0
            soma2 = 0.0
            soma3 = 0.0
            temp_glob_adj = []
            temp_k = []
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            for adj in adj_volumes:
                #2
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
                    soma2 = soma2 - keq2
                    soma3 = soma3 + (keq2*(self.tz-altura))
                #2
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
            soma2 = soma2*(self.tz-volume_centroid[2])
            soma2 = -(soma2 + soma3)
            temp_k.append(soma)
            temp_glob_adj.append(self.map_vols_ic[volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                index = self.wells_n.index(volume)
                tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                if volume in self.wells_inj:
                    #3
                    self.b[self.map_vols_ic[volume]] += self.set_q[index] + soma2
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[volume]] += -self.set_q[index] + soma2
            #1
            else:
                #2
                self.b[self.map_vols_ic[volume]] += soma2
        #0
        self.trans_fine.FillComplete()

    def set_global_problem_vf(self):
        """
        transmissibilidade da malha fina por volumes finitos

        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)
        self.A_np = np.zeros((len(self.all_fine_vols), len(self.all_fine_vols)))
        self.b_np = np.zeros(len(self.all_fine_vols))

        self.trans_fine2 = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b2 = Epetra.Vector(std_map)
        for volume in self.all_fine_vols:
            #1
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if volume not in self.wells_d:
                #2
                temp_k, temp_glob_adj = self.mount_lines_1(volume, self.map_all_fine_vols)
                self.trans_fine2.InsertGlobalValues(global_volume, temp_k, temp_glob_adj)
                self.A_np[global_volume, temp_glob_adj] = temp_k
                if volume in self.wells_n:
                    #3
                    index = self.wells_n.index(volume)
                    if volume in self.wells_inj:
                        #4
                        self.b2[global_volume] += self.set_q[index]
                        self.b_np[global_volume] += self.set_q[index]
                    #3
                    else:
                        #4
                        self.b2[global_volume] += -self.set_q[index]
                        self.b_np[global_volume] += -self.set_q[index]
            #1
            else:
                index = self.wells_d.index(volume)
                self.trans_fine2.InsertGlobalValues(global_volume, [1.0], [global_volume])
                self.A_np[global_volume, global_volume] = 1.0
                self.b2[global_volume] = self.set_p[index]
                self.b_np[global_volume] = self.set_p[index]
        #0
        self.trans_fine2.FillComplete()

    def set_global_problem_vf_3(self):
        """
        transmissibilidade da malha fina excluindo os volumes com pressao prescrita
        obs: com funcao para obter dados dos elementos
        """

        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)
        for volume in self.all_fine_vols_ic - set(self.neigh_wells_d):
            #1
            gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            temp_k, temp_glob_adj = self.mount_lines_1(volume, self.map_vols_ic)
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                index = self.wells_n.index(volume)
                if volume in self.wells_inj:
                    #3
                    self.b[self.map_vols_ic[volume]] = self.set_q[index]
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[volume]] = -self.set_q[index]
        #0
        for volume in self.neigh_wells_d:
            #1
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            soma = 0.0
            temp_glob_adj = []
            temp_k = []
            for adj in adj_volumes:
                #2
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                # uni = self.unitary(direction)
                uni = self.funcoes[0](direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/float(abs(self.mi*np.dot(direction, uni)))
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

    def set_Pc(self):
        """
        seta a pressao nos volumes da malha grossa

        """

        for primal in self.primals:

            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id]

            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            value = self.Pc[primal_id]
            self.mb.tag_set_data(
                self.pc_tag,
                fine_elems_in_primal,
                np.repeat(value, len(fine_elems_in_primal)))

    def set_perm(self):

        """
        seta a permeabilidade dos volumes da malha fina
        """
        perms = []
        perm_tensor = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 10.0]

        for elem in self.all_fine_vols:
            self.mb.tag_set_data(self.perm_tag, elem, perm_tensor)

        # perm_tensor = [10.0,  0.0, 0.0,
        #                 0.0, 10.0, 0.0,
        #                 0.0,  0.0, 5.0]
        #
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
        #     perms.append(perm_tensor)
        #     self.mb.tag_set_data(self.perm_tag, volume, perm_tensor)
        #
        # perms = np.array(perms)
        # np.savez_compressed('perms_an', perms = perms)


        # for volume in self.all_fine_vols:
        #     k1 = random.randint(1, 1001)*(10**(-3))
        #     k2 = random.randint(1, 1001)*(10**(-3))
        #     k3 = random.randint(1, 1001)*(10**(-3))
        #
        #     perm_tensor = [k1, 0, 0,
        #                    0, k2, 0,
        #                    0, 0, k3]
        #     perms.append(perm_tensor)
        #     self.mb.tag_set_data(self.perm_tag, volume, perm_tensor)
        #
        # perms = np.array(perms)
        # np.savez_compressed('perms_an', perms = perms)


        # # permeabilidades em camadas z
        # cont = 0
        # elems = []
        #
        # val = int(self.nx/2.0)
        # gid1 = np.array([0, 0, 0])
        # gid2 = np.array([val, self.ny-1, self.nz-1])
        # dif =  gid2 - gid1 + np.array([1, 1, 1])
        # gids = []
        # for k in range(dif[2]):
        #     for j in range(dif[1]):
        #         for i in range(dif[0]):
        #             gid = gid1 + np.array([i, j, k])
        #             gid = gid[0] + gid[1]*self.nx + gid[2]*self.nx*self.ny
        #             gids.append(gid)
        #
        #
        # perm_tensor_1 = [1.0, 0.0, 0.0,
        #                 0.0, 1.0, 0.0,
        #                 0.0, 0.0, 1.0]
        #
        # perm_tensor_2 = [0.5, 0.0, 0.0,
        #                  0.0, 0.5, 0.0,
        #                  0.0, 0.0, 0.5]
        #
        #
        # k1 = 1.0
        # k2 = 0.5
        #
        # for volume in self.all_fine_vols:
        #     gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        #     if gid in gids:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_1)
        #         self.mb.tag_set_data(self.k_tag, volume, k1)
        #     else:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_2)
        #         self.mb.tag_set_data(self.k_tag, volume, k2)

        # sudo docker pull padmec/elliptic:1.0
        # k3 = 100.0
        # perm_tensor_3 = [100.0, 0.0, 0.0,
        #                  0.0, 100.0, 0.0,
        #                  0.0, 0.0, 100.0]
        #
        # k1 = 1.0
        # perm_tensor_1 = [1.0, 0.0, 0.0,
        #                  0.0, 1.0, 0.0,
        #                  0.0, 0.0, 1.0]
        #
        # k2 = 10.0
        # perm_tensor_2 = [k2, 0.0, 0.0,
        #                  0.0, k2, 0.0,
        #                  0.0, 0.0, k2]
        #
        # k4 = 1000.0
        # perm_tensor_4 = [k4, 0.0, 0.0,
        #                  0.0, k4, 0.0,
        #                  0.0, 0.0, k4]
        #
        # gid1 = np.array([0, 0, 0])
        # gid2 = np.array([14, 29, 14])
        # dif = gid2 - gid1 + np.array([1, 1, 1])
        #
        # gids1 = []
        # for k in range(dif[2]):
        #     for j in range(dif[1]):
        #         for i in range(dif[0]):
        #             gid = gid1 + np.array([i, j, k])
        #             gid = gid[0] + gid[1]*self.nx + gid[2]*self.nx*self.ny
        #             gids1.append(gid)
        #
        # gid1 = np.array([15, 0, 0])
        # gid2 = np.array([29, 29, 14])
        # dif = gid2 - gid1 + np.array([1, 1, 1])
        #
        # gids2 = []
        # for k in range(dif[2]):
        #     for j in range(dif[1]):
        #         for i in range(dif[0]):
        #             gid = gid1 + np.array([i, j, k])
        #             gid = gid[0] + gid[1]*self.nx + gid[2]*self.nx*self.ny
        #             gids2.append(gid)
        #
        # gid1 = np.array([0, 0, 14])
        # gid2 = np.array([14, 29, 29])
        # dif = gid2 - gid1 + np.array([1, 1, 1])
        #
        # gids3 = []
        # for k in range(dif[2]):
        #     for j in range(dif[1]):
        #         for i in range(dif[0]):
        #             gid = gid1 + np.array([i, j, k])
        #             gid = gid[0] + gid[1]*self.nx + gid[2]*self.nx*self.ny
        #             gids3.append(gid)
        #
        #
        # for volume in self.all_fine_vols:
        #     gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
        #     if gid_vol in gids1:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_1)
        #         self.mb.tag_set_data(self.k_tag, volume, k1)
        #     elif gid_vol in gids2:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_2)
        #         self.mb.tag_set_data(self.k_tag, volume, k2)
        #     elif gid_vol in gids3:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_3)
        #         self.mb.tag_set_data(self.k_tag, volume, k3)
        #     else:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_4)
        #         self.mb.tag_set_data(self.k_tag, volume, k4)




        # for k in range(self.nz):
        #     gids = []
        #     gid1 = np.array([0, 0, k])
        #     gid2 = np.array([self.nx-1, self.ny-1, k])
        #     dif = (gid2 - gid1) + np.array([1, 1, 1])
        #     perm = random.randint(1, 999)*(10**(-3))
        #     for l in range(dif[2]):
        #         for m in range(dif[1]):
        #             for n in range(dif[0]):
        #                 gid = gid1 + np.array([n, m, l])
        #                 global_id = gid[0] + gid[1]*self.nx + gid[2]*self.nx*self.ny
        #                 elem = self.map_gids_in_elems[global_id]
        #                 perm_tensor = [perm, 0.0, 0.0,
        #                                 0.0, perm, 0.0,
        #                                 0.0, 0.0, perm]
        #                 perms.append(perm_tensor)
        #                 self.mb.tag_set_data(self.perm_tag, elem, perm_tensor)
        # perms = np.array(perms)
        # np.savez_compressed('perms', perms = perms)

        # carregar de um arquivo existente
        # perms = np.load('perms_an.npz')['perms']
        # i = 0
        # for elem in self.all_fine_vols:
        #     self.mb.tag_set_data(self.perm_tag, elem, perms[i])
        #     i += 1

    def set_perm_2(self):
        """
        seta a permeabilidade dos volumes da malha fina

        """

        perm_tensor_1 = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]
        #
        # perm_tensor_2 = [0.5, 0.0, 0.0,
        #                 0.0, 0.5, 0.0,
        #                 0.0, 0.0, 0.5]
        #
        # gid_lim = 323
        #
        # for volume in self.all_fine_vols:
        #     global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
        #     if global_volume <= gid_lim:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_1)
        #     else:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_2)

        # # direcao x
        # gid1 = np.array([0, 0, 0])
        # gid2 = np.array([4, 8, 8])
        # dif = (gid2 - gid1) + np.array([1, 1, 1])
        # gids = []
        #
        # for i in range(dif[0]):
        #     for j in range(dif[1]):
        #         for k in range(dif[2]):
        #             gid = gid1 + np.array([i, j, k])
        #             gids.append(gid[0] + self.nx*gid[1] + self.nx*self.ny*gid[2])
        #
        # for volume in self.all_fine_vols:
        #     global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
        #     if global_volume in gids:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_1)
        #     else:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_2)


        # gid1 = np.array([0, 0, 0])
        # gid2 = np.array([24, 12, 24])
        # dif = (gid2 - gid1) + np.array([1, 1, 1])
        # gids = []
        #
        # for i in range(dif[0]):
        #     for j in range(dif[1]):
        #         for k in range(dif[2]):
        #             gid = gid1 + np.array([i, j, k])
        #             gids.append(gid[0] + self.nx*gid[1] + self.nx*self.ny*gid[2])
        #
        # for volume in self.all_fine_vols:
        #     global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
        #     if global_volume in gids:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_1)
        #     else:
        #         self.mb.tag_set_data(self.perm_tag, volume, perm_tensor_2)

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

        std_map = Epetra.Map(n, 0, self.comm)

        x = Epetra.Vector(std_map)

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-9)

        return x

    def test_conservation_coarse(self):
        """
        verifica se o fluxo é conservativo nos volumes da malha grossa
        utilizando a pressao multiescala para calcular os fluxos na interface dos mesmos
        """

        #0
        lim = 10**(-6)
        soma = 0
        Qc2 = []
        prim = []
        for primal in self.primals:
            #1
            Qc = 0
            # my_adjs = set()
            primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id1]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_interface, volumes_in_primal = self.get_volumes_in_interfaces(
            fine_elems_in_primal, primal_id1, flag = 1)
            for volume in volumes_in_primal:
                #2
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                for adj in adjs_vol:
                    #3
                    if adj not in volumes_in_interface:
                        continue
                    # my_adjs.add(adj)
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
                    keq = self.kequiv(kvol, kadj)
                    keq = keq*(np.dot(self.A, uni))/(self.mi) #*np.dot(self.h, uni))
                    grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                    q = (grad_p)*keq
                    Qc += q
            #1
            # print('Primal:{0} ///// Qc: {1}'.format(primal_id, Qc))
            Qc2.append(Qc)
            prim.append(primal_id)
            self.mb.tag_set_data(self.flux_coarse_tag, fine_elems_in_primal, np.repeat(Qc, len(fine_elems_in_primal)))
            # if Qc > lim:
            #     print('Qc nao deu zero')
            #     import pdb; pdb.set_trace()
        with open('Qc.txt', 'w') as arq:
            for i,j in zip(prim, Qc2):
                arq.write('Primal:{0} ///// Qc: {1}\n'.format(i, j))
            arq.write('\n')
            arq.write('sum Qc:{0}'.format(sum(Qc2)))

    def test_operadores(self):
        mat = np.load('operadores.npz')
        OP = mat['OP']

        for i in OP:
            print(sum(i))
            # import pdb; pdb.set_trace()

    def unitary(self,l):
        """
        obtem o vetor unitario positivo da direcao de l

        """
        uni = l/np.linalg.norm(l)
        uni = uni*uni

        return uni

    def write_b(self):

        with open('b.txt', 'w') as arq:
            for i in range(self.nf):
                j = self.b[i]
                arq.write(str(j))
                arq.write('\n')

    def write_op(self, trilOP, nf, nc):

        OP = np.zeros((nf, nc))

        for i in range(nf):
            p = trilOP.ExtractGlobalRowCopy(i)
            OP[i, p[1]] = p[0]
            #print(sum(OP[i]))

        with open('OP.txt', 'w') as arq:
            for i in OP:
                for j in i:
                    arq.write(str(j))
                    arq.write('\n')

    def write_or(self, trilOR, nc, nf):

        OR = np.zeros((nc, nf))

        for i in range(nc):
            p = trilOR.ExtractGlobalRowCopy(i)
            OR[i, p[1]] = p[0]

        with open('OR.txt', 'w') as arq:
            for i in OR:
                for j in i:
                    arq.write(str(j))
                    arq.write('\n')

    def write_tf(self, trans_fine, nf):

        Tf = np.zeros((nf, nf))

        for i in range(nf):
            p = trans_fine.ExtractGlobalRowCopy(i)
            for j in range(len(p[1])):
                Tf[i, p[1][j]] = p[0][j]
            #Tf[i, p[1]] = p[0]

        with open('Tf.txt', 'w') as arq:
            for i in Tf:
                for j in i:
                    arq.write(str(j))
                    arq.write('\n')

    def obter_grafico(self):
        # # direcao z
        # l = []
        # for i in range(self.nz):
        #     gid = self.nx*self.ny*i
        #     l.append(gid)
        # pos = []
        # p = []
        # p2 = []
        # for volume in self.all_fine_vols:
        #     global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
        #     if global_volume in l:
        #         volume_centroid = self.mesh_topo_util.get_average_position([volume])
        #         value = self.mb.tag_get_data(self.pms_tag, volume, flat = True)[0]
        #         value2 = self.mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
        #         p.append(value)
        #         p2.append(value2)
        #         pos.append(volume_centroid[2])

        # direcao
        l = []
        for i in range(self.nx):
            gid = i
            l.append(gid)
        pos = []
        pf = []
        pms = []
        erro = []
        for volume in self.all_fine_vols:
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
            if global_volume not in l:
                continue
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            value = self.mb.tag_get_data(self.pms_tag, volume, flat = True)[0]
            value2 = self.mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            if volume in self.wells_d:
                erro.append(0.0)
            else:
                erro.append(100*abs((value2-value)/value2))
            pms.append(value)
            pf.append(value2)
            pos.append(volume_centroid[0]/float(self.nx))

        with open('grafico.txt', 'w') as arq:
            for i in range(len(pf)):
                arq.write('{0},{1},{2},{3}\n'.format(pos[i], pf[i], pms[i], erro[i]))

    def run(self):
        """
        substituicao direta dos valores de pressao prescrita
        """

        # Solucao direta
        self.set_global_problem_vf()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf)
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))

        # Solucao Multiescala
        self.calculate_prolongation_op_het()
        self.trilOP.FillComplete()

        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(
        self.trilOR, self.trans_fine, self.nf), self.trilOP, self.nf), self.nc, self.nc)

        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf, self.b), self.nc)
        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        self.Pms = self.multimat_vector(self.trilOP, self.nf, self.Pc)
        self.calculate_p_end()
        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))

        self.test_conservation_coarse()
        # self.Neuman_problem_6()
        # self.create_flux_vector()






        # #self.modificando_op()
        # #self.calculate_prolongation_op_het_2()
        # #self.calculate_prolongation_op_het_3()
        # #self.calculate_prolongation_op_het_4()
        # self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf)
        # self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))
        # self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf), self.trilOP, self.nf), self.nc, self.nc)
        # self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf, self.b), self.nc)
        # self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        # self.set_Pc(self.Pc)
        # self.Pms = self.multimat_vector(self.trilOP, self.nf, self.Pc)
        # #self.calculate_p_end()
        # self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))
        # self.Neuman_problem_4()
        # #self.calculate_pwf(self.pf_tag)
        # self.erro()
        #self.teste_numpy()


        #self.write_tf(self.trans_fine, self.nf)
        #self.write_op(self.trilOP, self.nf, self.nc)
        #self.write_or(self.trilOR, self.nc, self.nf)
        #self.write_b()


        self.mb.write_file('new_out_mono.vtk')

    def run_2(self):
        """
        eliminacao das linhas e colunas dos volumes com pressao prescrita
        """
        #0
        self.set_contorno()
        lim = 10**(-7)
        t1 = time.time()
        if self.flag_grav == 0:
            self.set_global_problem_vf_3()
        else:
            # self.set_global_problem_gr_vf_3()
            print('erro ____ sem gravidade')
            sys.exit(0)
        t10 = time.time()
        # #self.set_global_problem_gr_vf_2()

        # for i in range(self.nf_ic):
        #     p = self.trans_fine.ExtractGlobalRowCopy(i)
        #     soma = sum(p[0])
        #     print(i)
        #     print(p[0])
        #     print(soma)
        #     print('\n')
        #     import pdb; pdb.set_trace()

        # self.set_volumes_in_primal()

        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, len(self.all_fine_vols_ic))
        self.organize_Pf()
        del self.Pf
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf_all))
        del self.Pf_all
        #self.create_flux_vector_pf()
        # self.set_global_problem_vf()
        # self.Pf = self.solve_linear_problem(self.trans_fine2, self.b2, self.nf)
        # self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))



        # import pdb; pdb.set_trace()
        t2 = time.time()
        # self.set_global_problem_vf()
        # self.Pf2 = np.linalg.solve(self.A_np, self.b_np)
        # self.mb.tag_set_data(self.pf2_tag, self.all_fine_vols, np.asarray(self.Pf2))

        print('tempo solucao direta')
        print(t2 - t1)
        tempo_sol_direta = t2-t1
        # t2 = time.time()
        # print(' tempo Solucao direta')
        # print(t2-t1)
        # print('\n')

        t3 = time.time()
        if self.atualizar == 1:

            self.calculate_restriction_op_2()
            self.calculate_prolongation_op_het()
            # self.corretion_func_2()
            self.organize_op()
            #self.save_ops()

        else:
            self.load_ops()


        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf_ic), self.trilOP, self.nf_ic), self.nc, self.nc)

        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf_ic, self.b), self.nc)
        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        self.set_Pc()
        self.Pms = self.multimat_vector(self.trilOP, self.nf_ic, self.Pc)

        del self.trilOP
        del self.trilOR
        del self.Tc
        del self.Qc
        del self.Pc

        self.organize_Pms()
        del self.Pms
        # #self.add_gr_2()
        # #print(self.vect_gr)
        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms_all))
        del self.Pms_all
        #self.corretion_func()

        self.test_conservation_coarse()
        self.Neuman_problem_6()
        self.create_flux_vector_pms()

        t4 = time.time()
        self.erro()
        # print(' tempo Solucao Multiescala')
        # print(t3-t2)
        # print('\n')


        print('tempo solucao multiescala')
        print(t4-t3)
        tempo_sol_multiescala = t4-t3
        with open('tempo_de_simulacao.txt', 'w') as arq:
            arq.write('tempo_sol_direta:{0}\n'.format(tempo_sol_direta))
            arq.write('tempo_sol_multiescala:{0}\n'.format(tempo_sol_multiescala))
        #self.obter_grafico()
        #self.set_contorno()




        # with open('comparacao_de_fluxos.txt', 'w') as arq:
        #
        #     for volume in self.all_fine_vols:
        #         gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        #         flux_pms = self.store_flux[volume]
        #         flux_pf = self.store_flux_pf[volume]
        #         v1 = np.array([i for i in flux_pms.values()])
        #         v2 = np.array([i for i in flux_pf.values()])
        #         sum_pms = []
        #         sum_pf = []
        #         erro = []
        #         for i in range(len(v1)):
        #             erro.append(v2[i] - v1[i])
        #             sum_pms.append(v1[i])
        #             sum_pf.append(v2[i])
        #
        #         self.mb.tag_set_data(self.flux_fine_pms_tag, volume, sum(sum_pms))
        #         self.mb.tag_set_data(self.flux_fine_pf_tag, volume, sum(sum_pf))
        #
        #         # print('gid: {0}'.format(gid_vol))
        #         # print('flux_pms')
        #         # print(flux_pms)
        #         # print('flux_pf')
        #         # print(flux_pf)
        #         # print('erro')
        #         # print(erro)
        #         # print('\n')
        #         arq.write('gid:{0}\n'.format(gid_vol))
        #         arq.write('flux_pms\n')
        #         arq.write('{0}\n'.format(flux_pms))
        #         arq.write('sum flux pms\n')
        #         arq.write('{0}\n'.format(sum(sum_pms)))
        #         arq.write('sum flux pf\n')
        #         arq.write('{0}\n'.format(sum(sum_pf)))
        #         arq.write('flux_pf\n')
        #         arq.write('{0}\n'.format(flux_pf))
        #         arq.write('erro\n')
        #         arq.write('{0}\n'.format(erro))
        #         arq.write('\n')
        #         arq.write('\n')


        # self.corretion_func()

        #
        # # self.Neuman_problem_6()
        #
        #
        # # if self.flag_grav == 0:
        # #     self.Neuman_problem_4()
        # #     self.erro()
        # #     pass
        # # else:
        # #     self.Neuman_problem_4_gr_2()
        # #     self.erro()
        # #     pass
        #
        #
        # self.test_operadores()
        # self.erro_2()
        # #
        # # # self.erro_3()
        # # self.corretion_func()
        # # self.corretion_func_gr()
        # #self.add_gr()
        # #self.obter_grafico()
        # #print(Epetra.NumMyCols(self.trilOP))
        # #print(Epetra.NumMyRows(self.trilOP))
        # ##print(self.trilOP.NumMyRows())
        # ##print(self.trilOP.NumMyCols())


        self.mb.write_file('new_out_mono.vtk')

    def run_3(self):
        """
        imposicao de pressao na malha grossa
        """

        lim = 10**(-7)
        self.set_global_problem_vf()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf)
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))



        for volume in self.all_fine_vols:
            q = 0
            gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            for adj in adjs_vol:
                gid_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                pvol = self.mb.tag_get_data(self.pf_tag, volume, flat=True)[0]
                padj = self.mb.tag_get_data(self.pf_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)/(self.mi)
                grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                q += grad_p*(keq)*(np.dot(self.A, uni))
            if abs(q) > lim:
                print(gid_vol)
                print(q)
                import pdb; pdb.set_trace()





        # for primal in self.primals:
        #     #1
        #     Qc = 0
        #     primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
        #     primal_id = self.ident_primal[primal_id1]
        #     print('primal_id')
        #     print(primal_id)
        #     fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
        #     for volume in fine_elems_in_primal:
        #         global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        #         fine_to_primal = self.mb.tag_get_data(self.fine_to_primal_tag, volume, flat=True)[0]
        #         primal_vol = self.mb.tag_get_data(self.primal_id_tag, int(fine_to_primal), flat=True)[0]
        #         primal_vol = self.ident_primal[primal_vol]
        #         adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        #         print('global_volume: {0}'.format(global_volume))
        #         print('primal_vol: {0}'.format(primal_vol))
        #         print('\n')
        #
        #         adjs_volume = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        #         for adj in adjs_vol:
        #             global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
        #             fine_to_primal = self.mb.tag_get_data(self.fine_to_primal_tag, adj, flat=True)[0]
        #             primal_adj = self.mb.tag_get_data(self.primal_id_tag, int(fine_to_primal), flat=True)[0]
        #             primal_adj = self.ident_primal[primal_adj]
        #             print('global_adj: {0}'.format(global_adj))
        #             print('primal_adj: {0}'.format(primal_adj))
        #
        #         print('\n')
        #
        #         import pdb; pdb.set_trace()






        # # Solucao Multiescala
        # self.calculate_prolongation_op_het()
        # self.trilOP.FillComplete()
        #
        # self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(
        # self.trilOR, self.trans_fine, self.nf), self.trilOP, self.nf), self.nc, self.nc)
        #
        # self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf, self.b), self.nc)
        #
        # self.modificando_Tc_Qc()
        #
        # self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        # self.Pms = self.multimat_vector(self.trilOP, self.nf, self.Pc)
        # # self.calculate_p_end()
        # self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))

        # self.test_conservation_coarse()
        print('run_3')

        # self.erro()

        self.mb.write_file('new_out_mono.vtk')

    def run_4(self):
        """
        substitui 1 na linha do operador de prolongamento
        onde se tem volume com pressao prescrita
        """
        t1 = time.time()
        self.set_global_problem_vf()
        t2 = time.time()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf)
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))
        t3 = time.time()


        print('\n')


        # Solucao Multiescala
        self.calculate_prolongation_op_het()
        self.modificando_op()
        self.trilOP.FillComplete()

        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(
        self.trilOR, self.trans_fine, self.nf), self.trilOP, self.nf), self.nc, self.nc)

        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf, self.b), self.nc)

        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        self.Pms = self.multimat_vector(self.trilOP, self.nf, self.Pc)
        # self.calculate_p_end()
        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))

        t4 = time.time()

        print('tempo Solucao Multiescala')
        print((t4-t3) + (t2-t1))
        print('\n')
        print('tempo solucao direta')
        print(t2 - t1)

        # self.test_conservation_coarse()

        self.erro()


        self.mb.write_file('new_out_mono.vtk')

    def simulacoes(self):
        caminho3 = '/home/joao/simulacoes/999'
        caminho5 = '/home/joao/simulacoes/252525'
        principal1 = '/home/joao/git/local/Presto2/'
        principal = '/elliptic'
        caminho1 = '/elliptic/simulacoes'
        number_sim = 30
        caminho = caminho3
        print(os.getcwd())
        os.chdir(cm2)
        print(os.getcwd())
        # os.chdir(caminho)
        # arqs2 = os.listdir()
        # print(arqs2)
        # for n in range(number_sim):
        #     pasta = 'pasta{0}'.format(n)
