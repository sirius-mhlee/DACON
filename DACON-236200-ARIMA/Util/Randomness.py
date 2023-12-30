import os
import random
import numpy as np

import Config

os.environ['PYTHONHASHSEED'] = str(Config.seed)

random.seed(Config.seed)
np.random.seed(Config.seed)
rng = np.random.default_rng(Config.seed)
