import math
import os
import pickle
import random

import numpy as np
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

from inception_inference import load_model

dataset_use_percentage = 1
dataset_dir = 'mars32k'
dataset_files = list(os.listdir(dataset_dir))
random.shuffle(dataset_files)

dataset_files = dataset_files[:math.floor(len(dataset_files) * dataset_use_percentage)]
latent_vectors = np.zeros((len(dataset_files), 2048))

def get_batch(batch_size=1):
    for j in tqdm(range(0, len(dataset_files), batch_size)):
        batch = np.zeros((batch_size, 299, 299, 3))
        for i in range(batch_size):
            try:
                img = image.load_img(os.path.join(dataset_dir, dataset_files[j + i]), target_size=(299, 299), interpolation='bicubic')
            except IndexError:
                return
            batch[i] = image.img_to_array(img)
        yield batch, j, batch_size

model = load_model()

for batch, j, batch_size in get_batch(batch_size=64):
    x = preprocess_input(batch)
    latent_vectors[j:j+batch_size] = model.predict(x)

np.save('latent_vectors_' + dataset_dir, latent_vectors)
with open('./filelist.pickle', 'wb+') as f:
    pickle.dump(dataset_files, f)