import tensorflow as tf # Make sure to use tensorflow 2! 
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import math
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model


data_train, info = tfds.load("deep_weeds", with_info=True, split='train[:80%]')
data_valid  = tfds.load("deep_weeds",split='train[80%:90%]')
data_test = tfds.load("deep_weeds", split='train[90%:]')

NUM_CLASSES = 9

def prepare_weed_for_keras(dic):
  image = dic['image']
  preprocessed_image = tf.image.resize(image,[224,224])/255 # Scale to between 0 and 1
  label = dic['label']
  return preprocessed_image, label


def augment(image,label):
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  image = tf.image.random_hue(image, max_delta=0.1)
  image = tf.image.random_contrast(image, lower=0.7, upper=1.3) # Adjust contrast
  delta = tf.random.uniform([], math.radians(-360),math.radians(360))
  image = tfa.image.rotate(image, delta)
  return image, label

data_train_gen = (data_train
                  .map(prepare_weed_for_keras)
                  .cache()
                  .map(augment,
                     num_parallel_calls = tf.data.experimental.AUTOTUNE)
                  .batch(32)
                  .prefetch(tf.data.experimental.AUTOTUNE))
                  
data_valid_gen = (data_valid
                  .map(prepare_weed_for_keras)
                  .cache()
                  .batch(32))
                  
data_test_gen = data_test.map(prepare_weed_for_keras).batch(32)


image_input = Input((224,224,3))
resnet = ResNet50(include_top=False, weights="imagenet")
model = tf.keras.Sequential(
    [image_input,
     resnet,
     GlobalAveragePooling2D(),
     Dense(NUM_CLASSES, activation="softmax")])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


model.fit(data_train_gen,
          batch_size=32,
          epochs=50,
          validation_data=data_valid_gen)

print(model.evaluate(data_test_gen))
model.save("deep_weeds_resnet.h5")
