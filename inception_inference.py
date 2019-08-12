import os
from multiprocessing import Pool

from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

import imageio
import numpy as np
from matplotlib import pyplot as plt

#from comet_ml import Experiment
#import tensorflow as tf

#steps_per_epoch = 0
#total_steps = 0
#experiment = Experiment(api_key='<API_KEY>', project_name='latent clustering', workspace='schmidtdominik', log_code=False)


def load_model():
    base_model = InceptionV3(weights='imagenet')
    for l in base_model.layers:
        print(l.name, l)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    return model