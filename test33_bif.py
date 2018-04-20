import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import time
import sys

comm = Epetra.PyComm()
mb = core.Core()
mb.load_file('out.h5m')
root_set = mb.get_root_set()
mesh_topo_util = topo_util.MeshTopoUtil(mb)

class Msclassic_bif:
    def __init__(self):
        self.all_fine_vols = mb.get_entities_by_dimension(root_set, 3)
        self.nf = len(self.all_fine_vols)
        self.create_tags()
        self.read_structured()
        self.primals = mb.get_entities_by_type_and_tag(
                root_set, types.MBENTITYSET, np.array([self.primal_id_tag]),
                np.array([None]))
        self.nc = len(self.primals)
        self.ident_primal = []
        for primal in self.primals:
            primal_id = mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            self.ident_primal.append(primal_id)
        self.ident_primal = dict(zip(self.ident_primal, range(len(self.ident_primal))))
        self.loops = 40
        self.t = 10
        self.mi_w = 1.0
        self.mi_o = 1.3
        self.ro_w = 1.0
        self.ro_o = 0.9
        self.gama_w = 1.0
        self.gama_o = 1.0
        self.Swi = 0.2
        self.Swc = 0.2
        self.Sor = 0.2
        self.nw = 4
        self.no = 4
        self.gama_ = self.gama_w + self.gama_o
        self.set_k()
        self.set_fi()
        self.get_wells()
        self.calculate_restriction_op()
        self.read_perm_rel()

    def calculate_local_problem_het(self, elems, lesser_dim_meshsets, support_vals_tag):
        std_map = Epetra.Map(len(elems), 0, comm)
        linear_vals = np.arange(0, len(elems))
        id_map = dict(zip(elems, linear_vals))
        boundary_elms = set()

        b = Epetra.Vector(std_map)
        x = Epetra.Vector(std_map)

        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        for ms in lesser_dim_meshsets:
            lesser_dim_elems = mb.get_entities_by_handle(ms)
            for elem in lesser_dim_elems:
                if elem in boundary_elms:
                    continue
                boundary_elms.add(elem)
                idx = id_map[elem]
                A.InsertGlobalValues(idx, [1], [idx])
                b[idx] = mb.tag_get_data(support_vals_tag, elem, flat=True)[0]

        for elem in (set(elems) ^ boundary_elms):
            k_elem = mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
            lamb_w_elem = mb.tag_get_data(self.lamb_w_tag, elem)[0][0]
            lamb_o_elem = mb.tag_get_data(self.lamb_o_tag, elem)[0][0]
            centroid_elem = mesh_topo_util.get_average_position([elem])
            adj_volumes = mesh_topo_util.get_bridge_adjacencies(
                np.asarray([elem]), 2, 3, 0)
            values = []
            ids = []
            for adj in adj_volumes:
                if adj in id_map:
                    k_adj = mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
                    centroid_adj = mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_elem
                    uni = self.unitary(direction)
                    k_elem = np.dot(np.dot(k_elem,uni),uni)
                    k_elem = k_elem*(lamb_w_elem + lamb_o_elem)
                    k_adj = mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    k_adj = np.dot(np.dot(k_adj,uni),uni)
                    lamb_w_adj = mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                    lamb_o_adj = mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                    k_adj = k_adj*(lamb_w_adj + lamb_o_adj)
                    keq = self.kequiv(k_elem, k_adj)
                    keq = keq/(np.dot(self.h2, uni))
                    values.append(keq)
                    ids.append(id_map[adj])
                    k_elem = mb.tag_get_data(self.perm_tag, elem).reshape([3, 3])
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

        mb.tag_set_data(support_vals_tag, elems, np.asarray(x))

    def calculate_p_end(self):

        for volume in self.wells:
            global_volume = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if global_volume in self.wells_d:
                index = self.wells_d.index(global_volume)
                pms = self.set_p[index]
                mb.tag_set_data(self.pms_tag, volume, pms)

    def calculate_prolongation_op_het(self):

        zeros = np.zeros(self.nf)
        std_map = Epetra.Map(self.nf, 0, comm)
        self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        sets = mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, self.collocation_point_tag, np.array([None]))

        i = 0

        my_pairs = set()

        for collocation_point_set in sets:

            i += 1
            childs = mb.get_child_meshsets(collocation_point_set)
            collocation_point = mb.get_entities_by_handle(collocation_point_set)[0]
            primal_elem = mb.tag_get_data(self.fine_to_primal_tag, collocation_point,
                                           flat=True)[0]
            primal_id = mb.tag_get_data(self.primal_id_tag, int(primal_elem), flat=True)[0]

            primal_id = self.ident_primal[primal_id]

            support_vals_tag = mb.tag_get_handle(
                "TMP_SUPPORT_VALS {0}".format(primal_id), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)

            mb.tag_set_data(support_vals_tag, self.all_fine_vols, zeros)
            mb.tag_set_data(support_vals_tag, collocation_point, 1.0)

            for vol in childs:
                elems_vol = mb.get_entities_by_handle(vol)
                c_faces = mb.get_child_meshsets(vol)

                for face in c_faces:
                    elems_fac = mb.get_entities_by_handle(face)
                    c_edges = mb.get_child_meshsets(face)

                    for edge in c_edges:
                        elems_edg = mb.get_entities_by_handle(edge)
                        c_vertices = mb.get_child_meshsets(edge)
                        # a partir desse ponto op de prolongamento eh preenchido
                        self.calculate_local_problem_het(
                            elems_edg, c_vertices, support_vals_tag)

                    self.calculate_local_problem_het(
                        elems_fac, c_edges, support_vals_tag)

                self.calculate_local_problem_het(
                    elems_vol, c_faces, support_vals_tag)


                vals = mb.tag_get_data(support_vals_tag, elems_vol, flat=True)
                gids = mb.tag_get_data(self.global_id_tag, elems_vol, flat=True)
                primal_elems = mb.tag_get_data(self.fine_to_primal_tag, elems_vol,
                                               flat=True)

                for val, gid in zip(vals, gids):
                    if (gid, primal_id) not in my_pairs:
                        if val == 0.0:
                            pass
                        else:
                            self.trilOP.InsertGlobalValues([gid], [primal_id], val)

                        my_pairs.add((gid, primal_id))

        self.trilOP.FillComplete()

    def calculate_restriction_op(self):

        std_map = Epetra.Map(self.nf, 0, comm)
        self.trilOR = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        for primal in self.primals:

            primal_id = mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id]
            restriction_tag = mb.tag_get_handle(
                            "RESTRICTION_PRIMAL {0}".format(primal_id), 1, types.MB_TYPE_INTEGER,
                            True, types.MB_TAG_SPARSE)

            fine_elems_in_primal = mb.get_entities_by_handle(primal)

            mb.tag_set_data(
                self.elem_primal_id_tag,
                fine_elems_in_primal,
                np.repeat(primal_id, len(fine_elems_in_primal)))

            gids = mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True)
            self.trilOR.InsertGlobalValues(primal_id, np.repeat(1, len(gids)), gids)

            mb.tag_set_data(restriction_tag, fine_elems_in_primal, np.repeat(1, len(fine_elems_in_primal)))

        self.trilOR.FillComplete()

        """for i in range(len(primals)):
            p = trilOR.ExtractGlobalRowCopy(i)
            print(p[0])
            print(p[1])
            print('\n')"""

    def calculate_sat(self):
        lim = 0.00001

        for volume in self.all_fine_vols:
            gid = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            div = self.div_upwind_2(volume, self.pf_tag)
            fi = mb.tag_get_data(self.fi_tag, volume)[0][0]
            sat1 = mb.tag_get_data(self.sat_tag, volume)[0][0]
            sat = sat1 + div*(self.delta_t/(fi*self.V))
            if abs(div) < lim or sat1 == (1 - self.Sor) or gid in self.wells_d:
                continue
                #sys.exit(0)
            if sat < 0 or sat > (1 - self.Sor):
                if sat < 0:
                    sat = 0.0
                else:
                    sat = 1 - self.Sor

            mb.tag_set_data(self.sat_tag, volume, sat)

    def cfl(self, fi, qmax):
        cfl = 1.0

        self.delta_t = cfl*(fi*self.V)/qmax

    def create_tags(self):
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

        self.p_tag = mb.tag_get_handle(
                        "P", 1, types.MB_TYPE_DOUBLE,
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
            grad_p = padj - pvol

            if grad_p > 0:
                keq = (lamb_w_adj*kadj*(np.dot(self.A, uni)))/(np.dot(self.h, uni))

            else:
                keq = (lamb_w_vol*kvol*(np.dot(self.A, uni)))/(np.dot(self.h, uni))

            q = q + keq*grad_p
            kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

        return q

    def div_upwind_3(self, volume, p_tag):

        """
        a mobilidade da interface é dada pela media das mobilidades

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
            keq = self.kequiv(kvol, kadj)
            grad_p = padj - pvol

            lamb_eq = (lamb_w_vol + lamb_w_adj)/2.0
            keq = (keq*lamb_eq*(np.dot(self.A, uni)))/(np.dot(self.h, uni))

            q = q + keq*grad_p
            kvol = mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

        return q

    def get_wells(self):
        wells_d = []
        wells_n = []
        set_p = []
        set_q = []

        wells_set = mb.tag_get_data(self.wells_tag, 0, flat=True)[0]
        self.wells = mb.get_entities_by_handle(wells_set)

        for well in self.wells:
            global_id = mb.tag_get_data(self.global_id_tag, well, flat=True)[0]
            valor_da_prescricao = mb.tag_get_data(self.valor_da_prescricao_tag, well, flat=True)[0]
            tipo_de_prescricao = mb.tag_get_data(self.tipo_de_prescricao_tag, well, flat=True)[0]
            #raio_do_poco = mb.tag_get_data(raio_do_poco_tag, well, flat=True)[0]
            #tipo_de_poco = mb.tag_get_data(tipo_de_poco_tag, well, flat=True)[0]
            #tipo_de_fluido = mb.tag_get_data(tipo_de_fluido_tag, well, flat=True)[0]
            #pwf = mb.tag_get_data(pwf_tag, well, flat=True)[0]
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

    def kequiv(self, k1, k2):
        #keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
        keq = (2*k1*k2)/(k1+k2)

        return keq

    def modificar_matriz(self, A, rows, columns):

        row_map = Epetra.Map(rows, 0, comm)
        col_map = Epetra.Map(columns, 0, comm)

        C = Epetra.CrsMatrix(Epetra.Copy, row_map, col_map, 3)

        for i in range(rows):
            p = A.ExtractGlobalRowCopy(i)
            values = p[0]
            index_columns = p[1]
            C.InsertGlobalValues(i, values, index_columns)

        C.FillComplete()

        return C

    def modificar_vetor(self, v, nc):

        std_map = Epetra.Map(nc, 0, comm)
        x = Epetra.Vector(std_map)

        for i in range(nc):
            x[i] = v[i]


        return x

    def multimat_vector(self, A, row, b):

        std_map = Epetra.Map(row, 0, comm)
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
                            k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                            k_adj = np.dot(np.dot(k_adj,uni),uni)
                            keq = self.kequiv(k_vol, k_adj)
                            keq = keq/(np.dot(self.h2, uni))
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
                        keq = keq/(np.dot(self.h2, uni))
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
                self.mb.tag_set_data(self.p_tag, volume, x[i])
                self.mb.tag_set_data(self.pms2_tag, volume, x_np[i])

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

    def pol_interp_2(self, S):

        S_temp = (S - self.Swc)/(1 - self.Swc - self.Sor)
        krw = (S_temp)**(self.nw)
        kro = (1 - S_temp)**(self.no)

        return krw, kro

    def pymultimat(self, A, B, nf):

        nf_map = Epetra.Map(nf, 0, comm)

        C = Epetra.CrsMatrix(Epetra.Copy, nf_map, 3)

        EpetraExt.Multiply(A, False, B, False, C)

        C.FillComplete()

        return C

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
        a = np.array([ax, ay, az])

        hmin = min(hx, hy, hz)
        V = hx*hy*hz

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.h2 = h2
        self.h = h
        self.V = V
        self.A = a
        self.tz = tz

    def set_erro(self):
        for volume in self.all_fine_vols:
            Pf = mb.tag_get_data(self.pf_tag, volume, flat = True)[0]
            Pms = mb.tag_get_data(self.pms_tag, volume, flat = True)[0]
            erro = abs(Pf - Pms)/float(abs(Pf))
            mb.tag_set_data(self.err_tag, volume, erro)

    def set_fi(self):
        fi = 0.3
        for volume in self.all_fine_vols:
            mb.tag_set_data(self.fi_tag, volume, fi)

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

    def set_k(self):

        perm_tensor = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]

        for volume in self.all_fine_vols:
            mb.tag_set_data(self.perm_tag, volume, perm_tensor)

    def set_lamb(self):
        for volume in self.all_fine_vols:
            S = mb.tag_get_data(self.sat_tag, volume)[0][0]
            #print('S')
            #print(S)
            krw = self.pol_interp(S, self.Sw_r, self.krw_r)
            #print('krw')
            #print(krw)
            kro = self.pol_interp(S, self.Sw_r, self.kro_r)
            #print('kro')
            #print(kro)
            lamb_w = krw/self.mi_w
            lamb_o = kro/self.mi_o
            mb.tag_set_data(self.lamb_w_tag, volume, lamb_w)
            mb.tag_set_data(self.lamb_o_tag, volume, lamb_o)

    def set_lamb_2(self):
        for volume in self.all_fine_vols:
            S = mb.tag_get_data(self.sat_tag, volume)[0][0]
            krw, kro = self.pol_interp_2(S)
            lamb_w = krw/self.mi_w
            lamb_o = kro/self.mi_o
            mb.tag_set_data(self.lamb_w_tag, volume, lamb_w)
            mb.tag_set_data(self.lamb_o_tag, volume, lamb_o)

    def set_Pc(self):

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

        l = []
        for volume in self.wells:
            tipo_de_poco = mb.tag_get_data(self.tipo_de_poco_tag, volume)[0][0]
            if tipo_de_poco == 1:
                gid = mb.tag_get_data(self.global_id_tag, volume)[0][0]
                l.append(gid)


        for volume in self.all_fine_vols:
            gid = mb.tag_get_data(self.global_id_tag, volume)
            if gid in l:
                mb.tag_set_data(self.sat_tag, volume, 1.0)
            else:
                mb.tag_set_data(self.sat_tag, volume, 0.0)

    def solve_linear_problem(self, A, b, n):
        std_map = Epetra.Map(n, 0, comm)

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

    def unitary(self, l):
        uni = l/np.linalg.norm(l)
        uni = uni*uni

        return uni

    def run(self):

        t_ = 0.0
        loop = 0

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
        qmax, fi = self.div_max_2(self.pf_tag)
        self.cfl(fi, qmax)

        #calculo da pressao multiescala
        Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf), self.trilOP, self.nf), self.nc, self.nc)
        Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf, self.b), self.nc)
        self.Pc = self.solve_linear_problem(Tc, Qc, self.nc)
        self.set_Pc()
        self.Pms = self.multimat_vector(self.trilOP, self.nf, self.Pc)
        mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))
        self.calculate_p_end()
        self.set_erro()

        mb.write_file('new_out_bif{0}.vtk'.format(loop))

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
            t_ = t_ + self.delta_t
        """
