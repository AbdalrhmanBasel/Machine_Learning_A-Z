from easydict import EasyDict

from config.cfg import cfg
import numpy as np

np.random.seed(200)


class SportsmanHeight():

    def __call__(self):
        # The height is generated from a normal distribution with mean 160 and standard deviation 20
        football_player = np.random.randn(cfg.nb_football_player) * 20 + 160
        basketball_player = np.random.randn(cfg.nb_basketball_player) * 10 + 190

        dataset = {
            'height': np.concatenate((football_player, basketball_player)),
            'class': np.concatenate((np.zeros(cfg.nb_football_player),
                                     np.ones(cfg.nb_basketball_player))).astype(int)
        }

        return dataset





