from easydict import EasyDict

# Creating an instance of the EasyDict class and defining some configuration parameters
cfg = EasyDict()

cfg.nb_football_player = 500  # The number of football players to generate
cfg.nb_basketball_player = 500  # The number of basketball players to generate
cfg.max_height = 300  # The maximum height for both football and basketball players

# Dataset split
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.test_set_percent = 0.1

# estimator
cfg.threshold = 0.59
