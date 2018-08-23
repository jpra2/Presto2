from test34 import MsClassic_mono


class gravidade(MsClassic_mono):
    def __init__(self, ind = False):
        super().__init__(ind = ind)


sim_grav_mono = gravidade(ind = True)
