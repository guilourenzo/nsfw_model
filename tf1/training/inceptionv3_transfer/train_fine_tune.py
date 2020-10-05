import os
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD
from pathlib import Path
from keras.models import Sequential, Model, load_model
from datetime import datetime

print('Start: ', datetime.now())
# reusable stuff
import constants
import callbacks
import generators

# No kruft plz
clear_session()

# Config
height = constants.SIZES['basic']
width = height
weights_file = "weights.best_inception" + str(height) + ".hdf5"

print ('Starting from last full model run : ', datetime.now())
model = load_model("nsfw." + str(width) + "x" + str(height) + ".h5")

# Unlock a few layers deep in Inception v3
model.trainable = False
set_trainable = False
for layer in model.layers:
    if layer.name == 'conv2d_56':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Let's see it
print('Summary : ', datetime.now())
print(model.summary())

# Load checkpoint if one is found
if os.path.exists(weights_file):
        print ("loading ", weights_file)
        model.load_weights(weights_file)

# Get all model callbacks
callbacks_list = callbacks.make_callbacks(weights_file)

print('Compile model: ', datetime.now())
opt = SGD(momentum=.9)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Get training/validation data via generators
train_generator, validation_generator = generators.create_generators(height, width)

print('Start training! : ', datetime.now())
history = model.fit_generator(
    train_generator,
    callbacks=callbacks_list,
    epochs=constants.TOTAL_EPOCHS,
    steps_per_epoch=constants.STEPS_PER_EPOCH,
    shuffle=True,
    workers=4,
    use_multiprocessing=False,
    validation_data=validation_generator,
    validation_steps=constants.VALIDATION_STEPS
)

# Save it for later
print('Saving Model: ', datetime.now())
model.save("new_nsfw." + str(width) + "x" + str(height) + ".h5")
model.save("nsfw")

print('Finish: ', datetime.now())

final_loss, final_accuracy = model.evaluate(validation_generator, steps = constants.VALIDATION_STEPS)

print("Final loss: {:.2f}".format(final_loss))
print("Final accuracy: {:.2f}%".format(final_accuracy * 100))


import matplotlib as plt 

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()