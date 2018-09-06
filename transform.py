import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import tqdm

# Computed mean and std using preprocessing_function
MEGAASIAN_MEAN = np.array([ 0.09731393, -0.0889703 , -0.15700826])
MEGAASIAN_STD = np.array([0.45687136, 0.42319486, 0.4019396])


def zalo_preprocessing_function(x):
    # x = megaasian_crop(x)
    x = padding_image(x)
    x = resize(x, target_size=(128, 128))
    x = normalize(x)
    # x = mega_whitening(x, MEGAASIAN_MEAN, MEGAASIAN_STD)
    return x


def preprocessing_function(x):
    x = megaasian_crop(x)
    x = padding_image(x)
    x = resize(x, target_size=(128, 128))
    x = normalize(x)
    # x = mega_whitening(x, MEGAASIAN_MEAN, MEGAASIAN_STD)
    return x

def imdb_preprocessing_function(x):
    # if crop_type == 'imdb':
    x = imdb_crop(x)
    x = padding_image(x)
    x = resize(x, target_size=(160, 160))
    x = normalize(x)
    # x = mega_whitening(x, MEGAASIAN_MEAN, MEGAASIAN_STD)
    return x

def lap_preprocessing_function(x):
    # if crop_type == 'imdb':
    # x = imdb_crop(x)
    x = padding_image(x)
    x = resize(x, target_size=(160, 160))
    x = normalize(x)
    # x = mega_whitening(x, MEGAASIAN_MEAN, MEGAASIAN_STD)
    return x

def resize(x, target_size=(160, 160)):
    return cv2.resize(x, target_size)


def megaasian_crop(x):
    x = x[109-55:109+65, 89-45:89+45]
    return x


def imdb_crop(x):
    if len(x.shape) == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    h, w, c = x.shape
    x = x[int(h/2-h/3):int(h/2+h/4),
          int(w/2-w/4.5):int(w/2+w/4.5)]
    return x

    
def padding_image(x, padding_type = 'nearest'):
    """Preprocesses a numpy array encoding a batch of images.

    This function applies the "Inception" preprocessing which converts
    the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
    function is different from `imagenet_utils.preprocess_input()`.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
        padding_type: 
                    - default: zero padding
                    - white:   white padding (padding using values at 255)
                    - nearest: pad using nearest pixel intensity
    # Returns
        Preprocessed array.
    """
    old_h, old_w, _ = x.shape
    delta = 0
    top, bottom, left, right = 0, 0, 0, 0
    
    if old_w > old_h:
        delta = old_w - old_h
        top, bottom = delta//2, delta - (delta//2)
    else:
        delta = old_h - old_w
        left, right = delta//2, delta - (delta//2)
    color = [0, 0, 0]
    
    new_x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_REPLICATE, value=color)
    return new_x
#     return new_x


def label_normalize(x):
    if isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float32)
    print(x)
    x = x*1/99.0
    return x


def normalize(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x/255.0
    # x -= 1
    return x


def mega_whitening(x, mean, std):
    return np.multiply(np.subtract(x, mean), 1/std)


def computer_mean_std_with_paths(image_paths):
    ''' Compute global mean and std from image dataset 
        Parameters:
            image_paths: list of image paths 
        
        Return:
            mean: 3-D array 
            std:  3-D array
    '''
    print('There is/are {} paths in image_paths'.format(len(image_paths)))
    mean, std = [], [] # RGB 
    for p in tqdm.tqdm(image_paths):
        img = load_img(p)
        img = img_to_array(img)
        img = preprocessing_function(img)
        mean.append(np.mean(img, axis=(0, 1)))
        std.append(np.std(img, axis=(0, 1)))
    return [np.mean(mean, axis=0), np.mean(std, axis=0)]