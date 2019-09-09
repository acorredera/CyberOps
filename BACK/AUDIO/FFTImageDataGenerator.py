import numpy as np
import keras
from PIL import Image

#CODE ADAPTED FROM A detailed example of how to use data generators with Keras
#  (https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html) CHECKED ON 1/JULY/2018
class FFTImageDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels,imagesFolder,batch_size=32, dim=(224,224), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels = labels
        self.n_channels = n_channels
        self.imagesFolder = imagesFolder
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            path_img = "".join([self.imagesFolder, ID])
            img = Image.open(path_img)
            imgAsArray = np.asarray(img)
            X[i,] = imgAsArray  #np.load('data/' + ID + '.npy')

            # Store label
            y[i] = self.labels[ID]

        return X, y