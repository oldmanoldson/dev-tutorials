#Save and restore models
#Model progress can be saved during -  and after - training
#Setup
#install and imports
#Get an example dataset
import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

#Define a model
#Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation = tf.nn.relu, input_shape = (784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation = tf.nn.softmax)
    ])
    model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])
    return model

model = create_model()
model.summary()

#Save checkpoints during training
#Checkpoint callback usage
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)

model = create_model()

model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels), callbacks = [cp_callback]) #pass callback to training

#Now build a fresh, untrained model, and evaluate it on the test set
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

#Load the weights from the checkpoint and re-evaluate
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored Model, accuracy: {:5.2f}%".format(100 * acc))

#Checkpoint callback options
#include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose = 1, save_weights_only = True, period = 5) #Save weights, every 5-epochs

model = create_model()
model.fit(train_images, train_labels, epochs = 50, callbacks = [cp_callback], validation_data = (test_images, test_labels), verbose = 0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

#Test, reset the model and load the latest checkpoint
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored Model, accuracy: {:5.2f}%".format(100 * acc))

#Manually save weights
#Save the weights
model.save_weights('./checkpoints/my_checkpoint')

#Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#Save the entire model
model = create_model()

model.fit(train_images, train_labels, epochs = 5)

#Save entire model to a HDF5 file
model.save('my_model.h5')

#Recreate the exact same model, including weights and optimizer
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

#Check its accuracy
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#This technique saves everything:  the weight values, the model's configuration (architecture), the optimizer configuration
