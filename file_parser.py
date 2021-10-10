# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from genericpath import isdir
import struct
from struct import unpack
from os import makedirs, listdir
from os.path import join, exists, isfile

from h5py._hl import dataset
import constants as cnst
from skimage.io import imsave
import bs_lib.bs_string as bs
import bs_lib.bs_eda as beda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import h5py

def as_filename(id, with_category, extension='jpeg'):
    return '{:05d}_{}.{}'.format(id, with_category, extension)


def get_category_name(from_filename):
    name = bs.extract_string(from_filename, at_position=0, separator='.')
    category = bs.extract_string(from_string=name, at_position=3)
    return category

def get_images(from_image_path, nb_to_get=None,image_shape=None):
    images = np.load(from_image_path)
    if nb_to_get:
        idx = np.random.choice(images.shape[0], nb_to_get, replace=False)
        images = images[idx]
    if isinstance(image_shape,tuple):
        images = images.reshape(image_shape)
    return images

def get_rows_from(npy_file_path, nb_random_rows=None):
    array = np.load(npy_file_path)
    if nb_random_rows:
        idx = np.random.choice(array.shape[0], nb_random_rows, replace=False)
        array = array[idx]
    return array

def get_random_images(from_image_path, nb_to_get, image_shape=(-1, 28, 28)):
    """Get nb random choosed image from a .npy files

    Args:
        from_image_path (string): path to the .npy file
        nb_to_get (int): Number of sample to get
        image_shape(tuple,int): the shape of each image

    Returns:
        np.ndarray: an array of images as ndarray
    """
    images = np.load(from_image_path)
    idx = np.random.choice(images.shape[0], nb_to_get, replace=False)
    images = images[idx]
    images = images.reshape(image_shape)
    return images

def save_as_binary(array,as_filename,  into_directory, verbose=False):
    if not exists(into_directory):
        if verbose:
            print(f"Make directory: {into_directory}")
        makedirs(into_directory)

    if verbose:
        print(f"Saving array into {into_directory}")

    np.save(join(into_directory,as_filename),arr=array)
    if verbose:
        print(f"Array was saved as {as_filename} in directory: {into_directory}")


def save_as_jpeg(images, as_category, into_directory, verbose=False):
    """Create {into_directory}/{as_category}/ a jpeg image for each image in images, 
    rename using {number}_{as_category}.jpegself.
    Directories are created if they don't already exist 

    Args:
        images (numpy.ndarray): A 2d array of images 
        as_category (string): name of the category, it will be used to name directory and .jpeg files
        into_directory (string): path to the directory where files must be saved
    """
    category_directory_path = join(into_directory, as_category)

    if not exists(category_directory_path):
        if verbose:
            print(f"Make directory: {as_category} into {into_directory}")
        makedirs(category_directory_path)

    if verbose:
        print(f"Saving images into {category_directory_path}")

    count = 0
    for i in range(images.shape[0]):
        image = images[i]
        imsave(fname=join(category_directory_path, as_filename(
            i, as_category)), arr=image, check_contrast=False)
        count = i

    if verbose:
        print(f"{count} images were saved in {as_category} directory")

def merge_files(files):
    """Stack all files received. Files are npy files.

    Args:
        files (object): {"filename_1":"path_to_filename_1",...,"filename_n":"path_to_filename_n"}

    Returns:
        ndarray: merged arrays
    """    
    merged_array = []
    for filename, file in files.items():
        array = np.load(file)
        if len(merged_array)<1:
            merged_array=array
        else:
            merged_array = np.vstack(array)
    return merged_array

def create_dataset(dataset_name,from_directory_path, labels_dict, dest_directory_path=None, match_terms=[], exclude_terms=[], verbose=False):
    # takes all files needed from_directory_path
    files = beda.get_list_dir(from_directory_path, with_extension='.npy',match_terms=match_terms,exclude_terms=exclude_terms)
    # create destination path
    if dest_directory_path:
        dataset_file_path = join(dest_directory_path,dataset_name)
    else:
        dataset_file_path = join(from_directory_path,dataset_name)
    # write to h5 file
    with h5py.File(name=f"{dataset_file_path}.h5",mode="a") as f:
        # create empty dataset into h5
        f.create_dataset(name='image',shape=(0,784), maxshape=(None,784))
        f.create_dataset(name='category',shape=(0,),maxshape=(None,),dtype=int)

        for key,file in files.items():
            category=key
            for w in match_terms:
                category = category.replace(w,'')
            category = category.replace('_','')
            #load file content
            file_path = join(from_directory_path,file)
            images = np.load(file_path)
            # update dataset with data from files
            nb_images, _ = images.shape
            f["image"].resize((f["image"].shape[0] + nb_images), axis = 0)
            f["image"][-nb_images:] = images
            f["category"].resize((f["category"].shape[0] + nb_images), axis = 0)
            f["category"][-nb_images:] = np.full(nb_images,labels_dict[category])
    if verbose:
        print(f"File created @ {dataset_file_path}")
    return f"{dataset_file_path}.h5"

def load_dataset(dataset_filename,from_directory_path, verbose=False):
    if verbose:
        print(f"loading {dataset_filename} @ {from_directory_path}")
    file_path = join(from_directory_path,dataset_filename)
    X_image = None
    y_category = None
    if isfile(file_path):
        with h5py.File(file_path,'r') as file:
            X_image = file['image'][...]
            y_category = file['category'][...]
    if verbose:
        print("images: ",X_image.shape,"category:",y_category.shape)
    return X_image, y_category
        


def get_dataframe(from_directory, as_name, rewrite=False, match_terms=[],exclude_terms=[] ):
    file_path = join(from_directory, as_name)
    if not isfile(file_path) or rewrite:
        data = pd.DataFrame(columns=['filename', 'label'])
        # fetch if dataframe already exists
        files = beda.get_list_dir(from_directory,with_extension='.npy',match_terms=match_terms,exclude_terms=exclude_terms)
        for cat,file in files.items():
            temp = pd.DataFrame()
            directory_name = join(from_directory, cat)
            temp['filename'] = sorted(listdir(directory_name))
            temp['label'] = cat
            data = data.append(temp)
        data = data.reset_index()
        data = data.drop(columns='index')
        data.to_pickle(file_path)
    return pd.read_pickle(file_path)


def get_image_path(from_directory, with_category, with_id):
    filename = as_filename(with_id, with_category)
    return join(from_directory, with_category, filename)


def get_samples(from_directory, nb_samples=2):
    """Get nb_samples images for each category (as subfolder) from_directory

    Args:
        from_directory (string): Directory to search into
        nb_samples (int, optional): number of sample of each category to fetch. Defaults to 2.

    Returns:
        array, list: list of images path, list of images categories
    """
    images = []
    images_categories = []
    for category in listdir(from_directory):
        category_path = join(from_directory, category)
        if isdir(category_path):
            category_directory_path = listdir(category_path)
            for i in np.random.randint(0, len(category_directory_path), nb_samples):
                images.append(get_image_path(from_directory,
                              with_category=category, with_id=i))
                images_categories.append(category)
    return images, images_categories


def plot_images(images, images_categories, verbose=False):
    plt.figure(figsize=(13, 7))
    nb_images = len(images)
    nrows, ncols = beda.get_nrows_ncols(nb_images)
    if verbose:
        print(f"Prepare plots of {nb_images} images")
        print(f"Creating a grid {nrows}x{ncols}")
    for i in range(nb_images):
        if isinstance(images_categories,str):
            title = images_categories
        elif isinstance(images_categories,list):
            title = images_categories[i]
        else:
            title = i
        plt.subplot(nrows, ncols, i+1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], vmin=0, vmax=255, cmap='Greys')
        #plt.imshow(plt.imread(images[i]), vmin=0, vmax=255, cmap='Greys')
    plt.show()


def train_val_test_split(images, category, train_size=.75, val_size=.15, test_size=.1, verbose=False):
    if isinstance(category,str):
        y = np.full(images.shape[0], fill_value=category)
    else:
        y = category

    X_train, X_val, X_test, y_train, y_val, y_test = beda.train_val_test_split(
        images, y, 
        test_size=test_size, 
        train_size=train_size, 
        val_size=val_size, 
        random_state=1,
        show=verbose)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_outlier_mask(images, verbose=False):
    """Create a mask to mark outlier

    Args:
        images (ndarray): array of images

    Returns:
        ndarray: array of bool
    """    
    if verbose:
        print(f"Training on {images.shape[0]} images")
    # Isolation Forest to remove outlier
    model = IsolationForest(contamination='auto', n_estimators=100,random_state=0,n_jobs=-1)
    
    model.fit(images)
    if verbose:
        print("Predicting outliers")
    pred = model.predict(images)
    # crée un masque qui marque à True les anomalies
    mask = (pred==-1)
    # On applique le masque
    #outliers= images[mask]
    #outliers.shape # ou mask.sum() pour connaître le nombre d'anomalies trouvées
    if verbose:
        print(f"Mask done, outliers found: {mask.sum()}")
    return mask

def remove_outlier(images, verbose=False):
    mask = get_outlier_mask(images, verbose=verbose)
    filtered_images = images[~mask]
    if verbose:
        print(f'Nb images after removing outliers: {filtered_images.shape[0]}')
    return filtered_images

if __name__ == "__main__":
    pass
