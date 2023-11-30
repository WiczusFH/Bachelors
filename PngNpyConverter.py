from PIL import Image
import numpy as np
from PrincetonObjectIterator import Iterator
from Models import Config
import os

def save_image_as_npy(image_path, save_path):
    with Image.open(image_path) as img:
        img_array = np.array(img)
        np.save(save_path, img_array)

def get_flat_path(filename):
    return os.path.join(Config.Y_FOLDER,filename)

def iterator_action(source_path, target_path, filename):
    save_image_as_npy(source_path, target_path.replace(Config.PNG_FORMAT_NAME, Config.NPY_FORMAT_NAME))


iterator = Iterator(Config.RENDERED_FOlDER, Config.X_FOLDER, flatten=True)

iterator.execute(iterator_action, lambda x: x.endswith(Config.PNG_FORMAT_NAME))