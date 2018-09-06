import tensorflow as tf
from config import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


# TFdata_generator
class ImageGeneratorExtended(ImageDataGenerator):
    def load(self, images, labels, is_training, preprocessing_fn=None, batch_size=128):
        '''Construct a data generator using tf.Dataset'''


        def python_preprocess_fn(x):
            # Write preprocessing function here

            # If images is path then load it as image data
            # if self.is_path:
            x = load_img(x)
            x = img_to_array(x)
            if preprocessing_fn is not None:
                x = preprocessing_fn(x)
            if is_training:
                x = self.random_transform(x, seed=100)
            return x


        def preprocess_fn(image, label):
            '''A transformation function to preprocess raw data
            into trainable input. '''
            image = tf.cast(image, tf.float32)

            # Python wrapper to process input image
            image = tf.py_func(python_preprocess_fn, [image], [tf.float32])
            image = tf.reshape(image, (160, 160, 3))
            image = tf.cast(image, tf.float32)
            y = tf.cast(label, tf.float32)
            # label = float(label)
            return image, label

        # Convert images and labels into Tensor
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Shuffle if training
        if is_training:
            dataset = dataset.shuffle(1000)  # depends on sample size

        # Transform and batch data at the same time
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            preprocess_fn, batch_size,
            num_parallel_batches=64,  # cpu cores
            drop_remainder=True if is_training else False))
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset.make_one_shot_iterator()
