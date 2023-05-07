from easydict import EasyDict

cfg = EasyDict()

# Dataset path
# cfg.dataframe_path = 'advertising_data.csv'

# Configurations
cfg.learning_rate = 0.8
cfg.n_iterations = 5000
cfg.reg_coeff = 0

# Dataset size
cfg.train_size = 0.8
cfg.valid_size = 0.1
cfg.test_size = 0.1


# Base Functions
cfg.base_functions = [  # TODO list of basis functions
    lambda x: 1,
    lambda x: x,
    lambda x: x ** 2,
]
