import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import time
import math


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
        self.get_wells()
        self.set_perm()
        self.ro = 1.0
        self.mi = 1.0
        self.gama = 1.0
        self.nf = len(self.all_fine_vols)
        self.nc = len(self.primals)


        if ind == False:
            self.calculate_restriction_op()
            self.run()

        else:

            self.neigh_wells_d = [] #volumes da malha fina vizinhos as pocos de pressao prescrita
            self.elems_wells_d = [] #elementos com pressao prescrita
            for volume in self.wells:

                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                if global_volume in self.wells_d:

                    self.elems_wells_d.append(volume)

                    adjs_volume = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                    for adj in adjs_volume:

                        global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                        if global_adj not in self.wells_d:

                            self.neigh_wells_d.append(adj)

            self.all_fine_vols_ic = set(self.all_fine_vols) - set(self.elems_wells_d) #volumes da malha fina que sao icognitas
            gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols_ic, flat=True)
            self.map_vols_ic = dict(zip(gids_vols_ic, range(len(gids_vols_ic))))
            self.map_vols_ic_2 = dict(zip(range(len(gids_vols_ic)), gids_vols_ic))
            self.nf_ic = len(self.all_fine_vols_ic)

            self.run_2()

    def add_gr(self):

        cont = 0
        for primal in self.primals:
            volumes_in_interface = []
            soma = 0
            soma2 = 0
            soma3 = 0
            temp_glob_adj = []
            temp_k = []

            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            gids_in_primal = list(self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True))
            volumes_in_interface, id_gids = self.get_volumes_in_interfaces(primal_id, id_dict = True)
            std_map = Epetra.Map(len(id_gids), 0, self.comm)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            b = Epetra.Vector(std_map)
            #print(len(id_gids))
            A_np = np.zeros((len(id_gids), len(id_gids)))
            b_np = np.zeros(len(id_gids))

            for volume in fine_elems_in_primal:
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                if global_volume in self.wells_d:
                    index = self.wells_d.index(global_volume)
                    A.InsertGlobalValues(id_gids[global_volume], [1.0], [id_gids[global_volume]])
                    A_np[id_gids[global_volume], id_gids[global_volume]] = 1.0
                    b[id_gids[global_volume]] = self.set_p[index]
                    b_np[id_gids[global_volume]] = self.set_p[index]
                else:
                    volume_centroid = self.mesh_topo_util.get_average_position([volume])
                    adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
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

                    if global_volume in self.wells_n:
                        index = self.wells_n.index(global_volume)
                        tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                        if tipo_de_poco == 1:
                            b[id_gids[global_volume]] = self.set_q[index] + soma2
                            b_np[id_gids[global_volume]] = self.set_q[index] + soma2
                        else:
                            b[id_gids[global_volume]] = -self.set_q[index] + soma2
                            b_np[id_gids[global_volume]] = -self.set_q[index] + soma2
                    else:
                        b[id_gids[global_volume]] = soma2
                        b_np[id_gids[global_volume]] = soma2

                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

            A.FillComplete()
            x = self.solve_linear_problem(A, b, len(id_gids))
            x_np = np.linalg.solve(A_np, b_np)

            for volume in fine_elems_in_primal:
                global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                if global_volume in gids_in_primal:
                    #mb.tag_set_data(tag_da_gravidade, volume, x[global_volume])
                    pass
            """if cont == 0:
                print(x)
                print(x_np)"""
            cont += 1

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

        for volume in self.wells:
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if global_volume in self.wells_d:
                index = self.wells_d.index(global_volume)
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

        self.trilOP.FillComplete()

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

            WI = (2*math.pi*hz*c0)/(self.mi*np.log(rw/req))
            WI2 = (2*math.pi*hz*c0)/(self.mi*np.log(req/rw))

            tipo_de_prescricao = self.mb.tag_get_data(self.tipo_de_prescricao_tag, volume)[0][0]
            pvol = self.mb.tag_get_data(p_tag, volume)[0][0]

            if tipo_de_prescricao == 0:
                soma1 = 0.0
                soma2 = 0.0
                pvol = self.mb.tag_get_data(p_tag, volume)[0][0]
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
                    keq = keq/(np.dot(self.h2, uni))
                    soma1 = soma1 + keq
                    soma2 = soma2 + keq*padj
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

                soma1 = -soma1*pvol
                q = soma1 + soma2
                pwf1 = pvol - (q/(WI))
                pwf2 = pvol - (q/(WI2))

            else:
                q = self.mb.tag_get_data(self.valor_da_prescricao_tag, volume)[0][0]
                pwf1 = pvol - (q/(WI))
                pwf2 = pvol - (q/(WI2))
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
            gids_elems_ic = self.mb.tag_get_data(self.global_id_tag, elems_ic, flat=True)
            local_map = []
            for gid in gids_elems_ic:
                #2
                local_map.append(self.map_vols_ic[gid])
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

    def create_tags(self, mb):
        self.Pc2_tag = mb.tag_get_handle(
                        "PC2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pf2_tag = mb.tag_get_handle(
                        "PF2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.err_tag = mb.tag_get_handle(
                        "ERRO", 1, types.MB_TYPE_DOUBLE,
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
                        "PMS2", 1, types.MB_TYPE_DOUBLE,
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

        self.global_id_tag = mb.tag_get_handle("GLOBAL_ID")

        self.collocation_point_tag = mb.tag_get_handle("COLLOCATION_POINT")

        self.elem_primal_id_tag = mb.tag_get_handle(
            "FINE_PRIMAL_ID", 1, types.MB_TYPE_INTEGER, True,
            types.MB_TAG_SPARSE)

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

    def erro(self):
        for volume in self.all_fine_vols:
            Pf = self.mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            Pms = self.mb.tag_get_data(self.pms2_tag, volume, flat = True)[0]
            erro = abs(Pf - Pms)/float(abs(Pf))
            self.mb.tag_set_data(self.err_tag, volume, erro)

    def get_volumes_in_interfaces(self, primal_id, id_dict = False):

        """
        obtem uma lista com os elementos dos primais que estao na interface do primal_id

        se a flag id_dict == 1 retorna um mapeamento local da seguinte maneira:
        id_gids = dict(zip(gids_in_primal + gids_in_interface), range(len(gids_in_primal + gids_in_interface)))
        gids_in_primal == lista contendo os ids globais dos volumes dentro do respectivo primal
        gids_in_interface == lista contendo os ids globais dos volumes na interface do respectivo primal

        """

        volumes_in_interface = []
        gids_in_interface = []
        for primal in self.primals:
            id_primal = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            if id_primal == primal_id:
                fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
                gids_in_primal = list(self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True))

                for volume in fine_elems_in_primal:
                    global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                    adjs_volume = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)

                    for adj in adjs_volume:
                        fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, adj, flat=True)
                        primal_adj = self.mb.tag_get_data(
                            self.primal_id_tag, int(fin_prim), flat=True)[0]

                        if primal_adj != primal_id:
                            volumes_in_interface.append(adj)
                            global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                            gids_in_interface.append(global_adj)

                local_id = gids_in_primal + gids_in_interface
                local_id = dict(zip(local_id, range(len(local_id))))

                if id_dict == True:
                    return volumes_in_interface, local_id
                else:
                    return volumes_in_interface

            else:
                continue

    def get_wells(self):
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

        wells_set = self.mb.tag_get_data(self.wells_tag, 0, flat=True)[0]
        self.wells = self.mb.get_entities_by_handle(wells_set)
        wells = self.wells

        for well in wells:
            global_id = self.mb.tag_get_data(self.global_id_tag, well, flat=True)[0]
            valor_da_prescricao = self.mb.tag_get_data(self.valor_da_prescricao_tag, well, flat=True)[0]
            tipo_de_prescricao = self.mb.tag_get_data(self.tipo_de_prescricao_tag, well, flat=True)[0]
            #raio_do_poco = self.mb.tag_get_data(self.raio_do_poco_tag, well, flat=True)[0]
            #tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, well, flat=True)[0]
            #tipo_de_fluido = self.mb.tag_get_data(self.tipo_de_fluido_tag, well, flat=True)[0]
            #pwf = self.mb.tag_get_data(self.pwf_tag, well, flat=True)[0]
            if tipo_de_prescricao == 0:
                wells_d.append(global_id)
                set_p.append(valor_da_prescricao)
            else:
                wells_n.append(global_id)
                set_q.append(valor_da_prescricao)


        self.wells_d = wells_d
        self.wells_n = wells_n
        self.set_p = set_p
        self.set_q = set_q

    def kequiv(self,k1,k2):
        """
        obbtem o k equivalente entre k1 e k2

        """
        #keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
        keq = (2*k1*k2)/(k1+k2)

        return keq

    def modificando_op(self):

        for volume in self.wells:
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if global_volume in self.wells_d:
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

        """for volume in self.wells:
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if global_volume in self.wells_d:
                p = self.trilOP.ExtractGlobalRowCopy(global_volume)
                print(global_volume)
                print(p[1])
                print(p[0])"""

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
        Recalcula as presssoes em cada volume da seguinte maneira:
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

            volumes_in_interface = []#v1
            volumes_in_primal = []#v2
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            #setfine_elems_in_primal = set(fine_elems_in_primal)

            for fine_elem in fine_elems_in_primal:

                global_volume = self.mb.tag_get_data(self.global_id_tag, fine_elem, flat=True)[0]
                volumes_in_primal.append(fine_elem)
                adj_fine_elems = self.mesh_topo_util.get_bridge_adjacencies(fine_elem, 2, 3)

                for adj in adj_fine_elems:
                    fin_prim = self.mb.tag_get_data(self.fine_to_primal_tag, adj, flat=True)
                    primal_adj = self.mb.tag_get_data(
                        self.primal_id_tag, int(fin_prim), flat=True)[0]

                    if primal_adj != primal_id:
                        volumes_in_interface.append(adj)

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
                centroid_volume = self.mesh_topo_util.get_average_position([volume])
                k_vol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                adj_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)

                if volume in self.wells:
                    tipo_de_prescricao = self.mb.tag_get_data(self.tipo_de_prescricao_tag, volume)[0][0]
                    if tipo_de_prescricao == 0:
                        valor_da_prescricao = self.mb.tag_get_data(self.valor_da_prescricao_tag, volume)[0][0]
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
                            k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                            k_adj = np.dot(np.dot(k_adj,uni),uni)
                            #keq = self.kequiv(k_vol, k_adj)
                            keq = self.kequiv(k_vol, k_adj)
                            keq = keq*(np.dot(self.A, uni)*self.ro)/(self.mi*np.dot(self.h, uni))
                            soma = soma + keq
                            temp_k.append(keq)
                            temp_id.append(id_map[adj])
                        soma = -1*soma
                        temp_k.append(soma)
                        temp_id.append(id_map[volume])
                        tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                        valor_da_prescricao = self.mb.tag_get_data(self.valor_da_prescricao_tag, volume)[0][0]
                        if tipo_de_poco == 1:
                            b[id_map[volume]] = -valor_da_prescricao
                            b_np[id_map[volume]] = -valor_da_prescricao
                        else:
                            b[id_map[volume]] = valor_da_prescricao
                            b_np[id_map[volume]] = valor_da_prescricao

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
                        #keq = keq/(np.dot(self.h2, uni))
                        keq = keq*(np.dot(self.A, uni)*self.ro)/(self.mi*np.dot(self.h, uni))
                        soma = soma + keq
                        temp_k.append(keq)
                        temp_id.append(id_map[adj])
                    soma = -1*soma
                    temp_k.append(soma)
                    temp_id.append(id_map[volume])

                A.InsertGlobalValues(id_map[volume], temp_k, temp_id)
                A_np[id_map[volume], temp_id] = temp_k[:]

            x = self.solve_linear_problem(A, b, dim)
            x_np = np.linalg.solve(A_np, b_np)

            for i in range(len(volumes_in_primal) - len(volumes_in_interface)):
                volume = volumes_in_primal[i]
                self.mb.tag_set_data(self.pcorr_tag, volume, x[i])
                self.mb.tag_set_data(self.pms2_tag, volume, x_np[i])

    def organize_op(self):
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        trilOP2 = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
        gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols_ic, flat=True)
        cont = 0
        for i in gids_vols_ic:
            #1
            p = self.trilOP.ExtractGlobalRowCopy(i)
            values = p[0]
            index = p[1]
            trilOP2.InsertGlobalValues(self.map_vols_ic[i], list(values), list(index))
        #0
        self.trilOP = trilOP2

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
        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)
        Pf2 = Epetra.Vector(std_map)
        for i in range(len(self.Pf)):
            #1
            value = self.Pf[i]
            ind = self.map_vols_ic_2[i]
            Pf2[ind] = value
        #0
        for i in range(len(self.wells_d)):
            #1
            value = self.set_p[i]
            ind = self.wells_d[i]
            Pf2[ind] = value
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
            ind = self.map_vols_ic_2[i]
            Pms2[ind] = value
        #0
        for i in range(len(self.wells_d)):
            #1
            value = self.set_p[i]
            ind = self.wells_d[i]
            Pms2[ind] = value
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
        tx = int(a[0].strip())
        ty = int(a[1].strip())
        tz = int(a[2].strip())

        hx = tx/float(nx)
        hy = ty/float(ny)
        hz = tz/float(nz)
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
        self.A = A
        self.V = V

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

    def set_global_problem_vf(self):
        """
        transmissibilidade da malha fina por volumes finitos

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
                    keq = keq*(np.dot(self.A, uni)*self.ro)/(self.mi*np.dot(self.h, uni))

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
                        self.b[global_volume] = self.set_q[index]*V
                    else:
                        self.b[global_volume] = -self.set_q[index]*V

            else:
                index = self.wells_d.index(global_volume)
                self.trans_fine.InsertGlobalValues(global_volume, [1.0], [global_volume])
                self.b[global_volume] = self.set_p[index]

        self.trans_fine.FillComplete()

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
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni)*self.ro)/(self.mi*np.dot(self.h, uni))
                temp_glob_adj.append(self.map_vols_ic[global_adj])
                temp_k.append(-keq)
                soma = soma + keq
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            temp_k.append(soma)
            temp_glob_adj.append(self.map_vols_ic[global_volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[global_volume], temp_k, temp_glob_adj)
            if global_volume in self.wells_n:
                #2
                index = self.wells_n.index(global_volume)
                tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                if tipo_de_poco == 1:
                    #3
                    self.b[self.map_vols_ic[global_volume]] = self.set_q[index]*self.V
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[global_volume]] = -self.set_q[index]*self.V
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
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni)*self.ro)/(self.mi*np.dot(self.h, uni))
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
            temp_k.append(soma)
            temp_glob_adj.append(self.map_vols_ic[global_volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[global_volume], temp_k, temp_glob_adj)
            if global_volume in self.wells_n:
                #2
                index = self.wells_n.index(global_volume)
                tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                if tipo_de_poco == 1:
                    #3
                    self.b[self.map_vols_ic[global_volume]] += self.set_q[index]*V
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[global_volume]] += -self.set_q[index]*V
        #0
        self.trans_fine.FillComplete()

    def set_Pc(self, Pc):
        """
        seta a pressao nos volumes da malha grossa

        """

        for primal in self.primals:

            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id]

            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            value = Pc[primal_id]
            self.mb.tag_set_data(
                self.pc_tag,
                fine_elems_in_primal,
                np.repeat(value, len(fine_elems_in_primal)))

    def set_perm(self):
        """
        seta a permeabilidade dos volumes da malha fina

        """

        perm_tensor = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]

        for volume in self.all_fine_vols:
            self.mb.tag_set_data(self.perm_tag, volume, perm_tensor)

    def solve_linear_problem(self, A, b, n):

        std_map = Epetra.Map(n, 0, self.comm)

        x = Epetra.Vector(std_map)

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(1000, 1e-9)

        return x

    def teste_numpy(self):
        #0
        OP = np.zeros((self.nf_ic, self.nc))
        for i in range(self.nf_ic):
            #1
            p = self.trilOP.ExtractGlobalRowCopy(i)
            OP[i, p[1]] = p[0]
        #0
        OR = np.zeros((self.nc, self.nf_ic))
        for i in range(self.nc):
            #1
            p = self.trilOR.ExtractGlobalRowCopy(i)
            OR[i, p[1]] = p[0]
        #0
        Tf = np.zeros((self.nf_ic, self.nf_ic))
        for i in range(self.nf_ic):
            #1
            p = self.trans_fine.ExtractGlobalRowCopy(i)
            for j in range(len(p[1])):
                #2
                Tf[i, p[1][j]] = p[0][j]
        #0
        b = np.zeros(self.nf_ic)
        for i in range(self.nf_ic):
            #1
            b[i] = self.b[i]
        #0
        Tc = np.dot(OR, np.dot(Tf, OP))
        Qc = np.dot(OR, b)
        Pc = np.linalg.solve(Tc, Qc)
        Pms = np.dot(OP, Pc)
        Pms2 = np.zeros(len(self.all_fine_vols))
        for i in range(len(Pms)):
            #1
            value = Pms[i]
            ind = self.map_vols_ic_2[i]
            Pms2[ind] = value
        #0
        for i in range(len(self.wells_d)):
            #1
            value = self.set_p[i]
            ind = self.wells_d[i]
            Pms2[ind] = value
        #0
        self.Pms_all = Pms2

        """TfMs = np.dot(OP, np.dot(invTc, OR))
        temp = np.zeros(self.nf)
        for i in range(self.nf):
            if i in self.wells_d:
                TfMs[i,:] = temp.copy()
                TfMs[i,i] = 1.0
        Pms = np.linalg.solve(TfMs, b)
        mb.tag_set_data(self.pms3_tag, self.all_fine_vols, Pms)"""


        """with open('b.txt', 'w') as arq:
            for j in b:
                arq.write(str(j))
                arq.write('\n')

        with open('TfMs.txt', 'w') as arq:
            for i in TfMs:
                for j in i:
                    arq.write(str(j))
                    arq.write('\n')"""

        #Pm

    def unitary(self,l):
        """
        obtem o vetor unitario da direcao de l

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

    def run(self):


        #self.add_gr()

        #self.set_global_problem()
        #self.set_global_problem_gr()
        #self.set_global_problem_gr_vf()
        self.set_global_problem_vf()
        self.calculate_prolongation_op_het()
        #self.modificando_op()
        #self.calculate_prolongation_op_het_2()
        #self.calculate_prolongation_op_het_3()
        #self.calculate_prolongation_op_het_4()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf)
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))
        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf), self.trilOP, self.nf), self.nc, self.nc)
        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf, self.b), self.nc)
        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        self.set_Pc(self.Pc)
        self.Pms = self.multimat_vector(self.trilOP, self.nf, self.Pc)
        #self.calculate_p_end()
        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))
        self.Neuman_problem_4()
        #self.calculate_pwf(self.pf_tag)
        self.erro()
        #self.teste_numpy()


        #self.write_tf(self.trans_fine, self.nf)
        #self.write_op(self.trilOP, self.nf, self.nc)
        #self.write_or(self.trilOR, self.nc, self.nf)
        #self.write_b()


        self.mb.write_file('new_out_mono.vtk')

    def run_2(self):
        #0
        #help(Epetra.CrsMatrix)

        self.calculate_restriction_op_2()
        self.set_global_problem_vf_2()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, len(self.all_fine_vols_ic))
        self.organize_Pf()
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf_all))
        self.calculate_prolongation_op_het()
        self.organize_op()
        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf_ic), self.trilOP, self.nf_ic), self.nc, self.nc)
        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf_ic, self.b), self.nc)
        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        #self.Pms = self.multimat_vector(self.trilOP, self.nf_ic, self.Pc)
        #self.organize_Pms()
        self.teste_numpy()
        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms_all))
        self.Neuman_problem_4()
        self.erro()


        """print(Epetra.NumMyCols(self.trilOP))
        print(Epetra.NumMyRows(self.trilOP))"""
        #print(self.trilOP.NumMyRows())
        #print(self.trilOP.NumMyCols())














        self.mb.write_file('new_out_mono.vtk')
