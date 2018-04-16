import numpy as np
import time

from .StructuredMultiscaleMesh import StructuredMultiscaleMesh


class Preprocessor(object):
    """
    Creates a 3-D structured grid and aggregates elements in primal and dual
    coarse entities.
    """

    def __init__(self, configs):
        self.configs = configs

        self.structured_configs = self.configs['StructuredMS']
        self.coarse_ratio = self.structured_configs['coarse-ratio']
        self.mesh_size = self.structured_configs['mesh-size']
        self.block_size = self.structured_configs['block-size']

        self.info_p = self.configs['info_p']
        self.numero_de_pocos = int(self.info_p['numero-de-pocos'])

        d = []

        for i in range(self.numero_de_pocos):
            temp = []
            self.poco = self.configs['P{0}'.format(i)]
            poco = 'P{0}'.format(i)
            #temp.append(poco)
            localizacao = self.poco['localizacao']

            global_id = []
            for i in localizacao:
                global_id.append(int(i))
            temp.append(global_id)

            tipo_de_poco = int(self.poco['tipo-de-poco'])
            temp.append(tipo_de_poco)

            tipo_de_fluido = int(self.poco['tipo-de-fluido'])
            temp.append(tipo_de_fluido)

            tipo_de_prescricao = int(self.poco['tipo-de-prescricao'])
            temp.append(tipo_de_prescricao)

            valor_da_prescricao = float(self.poco['valor-da-prescricao'])
            temp.append(valor_da_prescricao)

            pwf = float(self.poco['pwf'])
            temp.append(pwf)

            raio_do_poco = float(self.poco['raio-do-poco'])
            temp.append(raio_do_poco)

            perfuracoes = int(self.poco['perfuracoes'])
            temp.append(perfuracoes)

            d.append(temp[:])

        self.wells = d[:]

        self.smm = StructuredMultiscaleMesh(
            self.coarse_ratio, self.mesh_size, self.block_size, self.wells)

    def run(self, moab):
        self.smm.set_moab(moab)

        self.smm.calculate_primal_ids()
        self.smm.create_tags()

        print("Creating fine vertices...")
        t0 = time.time()
        self.smm.create_fine_vertices()
        print("took {0}\n".format(time.time()-t0))

        print("Creating fine blocks and primal...")
        t0 = time.time()
        self.smm.create_fine_blocks_and_primal()
        print("took {0}\n".format(time.time()-t0))

        print("Generating dual...")
        t0 = time.time()
        self.smm.generate_dual()
        self.smm.store_primal_adj()
        print("took {0}\n".format(time.time()-t0))

        self.smm.create_wells()
        #self.smm.create_wells_2()

    @property
    def structured_configs(self):
        return self._structured_configs

    @structured_configs.setter
    def structured_configs(self, configs):
        if not configs:
            raise ValueError("Must have a [StructuredMS] section "
                             "in the config file.")

        self._structured_configs = configs

    @property
    def coarse_ratio(self):
        return self._coarse_ratio

    @coarse_ratio.setter
    def coarse_ratio(self, values):
        if not values:
            raise ValueError("Must have a coarse-ratio option "
                             "under the [StructuredMS] section in the config "
                             "file.")

        self._coarse_ratio = [int(v) for v in values]

    @property
    def mesh_size(self):
        return self._mesh_size

    @mesh_size.setter
    def mesh_size(self, values):
        if not values:
            raise ValueError("Must have a mesh-size option "
                             "under the [StructuredMS] section in the config "
                             "file.")

        self._mesh_size = [int(v) for v in values]

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, values):
        if not values:
            raise ValueError("Must have a block-size option "
                             "under the [StructuredMS] section in the config "
                             "file.")

        self._block_size = [int(v) for v in values]
