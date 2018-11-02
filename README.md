# Visualizing intermediate activation in Convolutional Neural Networks with Keras

Visualizing intermediate activations in Convolutional Neural Networks
In this article we're going to train a simple Convolutional Neural Network using Keras in Python for a classification task. For that we are going to use a very small and simple set of images consisting of 100 pictures of circles, 100 pictures of squares and 100 pictures of triangles which I took from Kaggle (https://www.kaggle.com/dawgwelder/keras-cnn-build/data). These will be split into training and testing sets (folders in working directory) and fed to the network.

Most importantly, we are going to replicate some of the work of François Chollet in his book Deep Learning with Python in order to learn how our layer structure processes the data in terms of visualization of each intermediate activation, which consists of displaying the feature maps that are output by the convolution and pooling layers in the network.

We'll go super fast since we are not focusing here on doing a full tutorial of CNNs with Keras but just this simple findings on how CNNs work.

Let's first import all our required libraries:

```
%matplotlib inline
​
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as im
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
These are our training images:
```

Circles

```
images = []
for img_path in glob.glob('training_set/circles/*.png'):
    images.append(mpimg.imread(img_path))
​
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)
```

Squares

```
images = []
for img_path in glob.glob('training_set/squares/*.png'):
    images.append(mpimg.imread(img_path))
​
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)
```

Triangles

```
images = []
for img_path in glob.glob('training_set/triangles/*.png'):
    images.append(mpimg.imread(img_path))
​
plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)
```

The shape of the images:

```
img = im.imread('training_set/squares/drawing(40).png')
img.shape
```

(28, 28, 3)
Images shapes are of 28 pixels by 28 pixels in RGB scale (although they are arguably black and white only)

Let's now proceed with our Convolutional Neural Network construction. As usually, we initiate the model with Sequential():


# Initialising the CNN

```
classifier = Sequential()
```

We specify our convolution layers and add MaxPooling to downsample and Dropout to prevent overfitting. We use Flatten and end with a Dense layer of 3 units, one for each class (circle [0], square [1], triangle [1]). We specify softmax as our last activation function, which is suggested for multiclass classification.


# Step 1 - Convolution

```
classifier.add(Conv2D(32, (3, 3), padding='same', input_shape = (28, 28, 3), activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25
​
# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25
​
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25
​
# Step 3 - Flattening
classifier.add(Flatten())
​
# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.5)) 
classifier.add(Dense(units = 3, activation = 'softmax'))
```

For this type of images, I might be building an overly complex structure, and that will be evident once we take a look at the feature maps, however, for the sake of this article, it helps me to showcase exactly what each layer will be doing. I'm certain we can obtain the same or better results with less layers and less complexity.

Let's take a look at our model summary:

```
classifier.summary()
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_25 (Conv2D)           (None, 28, 28, 32)        896       
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 26, 26, 32)        9248      
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 13, 13, 32)        0         
_________________________________________________________________
dropout_17 (Dropout)         (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 13, 13, 64)        18496     
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 11, 11, 64)        36928     
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 5, 5, 64)          0         
_________________________________________________________________
dropout_18 (Dropout)         (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 5, 5, 64)          36928     
_________________________________________________________________
conv2d_30 (Conv2D)           (None, 3, 3, 64)          36928     
_________________________________________________________________
max_pooling2d_15 (MaxPooling (None, 1, 1, 64)          0         
_________________________________________________________________
dropout_19 (Dropout)         (None, 1, 1, 64)          0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 64)                0         
_________________________________________________________________
dense_9 (Dense)              (None, 512)               33280     
_________________________________________________________________
dropout_20 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 3)                 1539      
=================================================================
Total params: 174,243
Trainable params: 174,243
Non-trainable params: 0
_________________________________________________________________
We compile the model utilizing rmsprop as our optimizer, categorical_crossentropy as our loss function and we specify accuracy as the metric we want to keep track of:


# Compiling the CNN
```
classifier.compile(optimizer = 'rmsprop',
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
```

## Using ImageDataGenerator to read images from directories
At this point we need to convert our pictures to a shape that the model will accept. For that we use the ImageDataGenerator. We initiate it and feed our images with .flow_from_directory. There are two main folders inside the working directory, called training_set and test_set. Each of those have 3 subfolders called circles, squares and triangles. I have sent 70 images of each shape to the training_set and 30 to the test_set.

```
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
​
training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (28, 28),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')
​
test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (28, 28),
                                            batch_size = 16,
                                            class_mode = 'categorical')
```

## Utilize callback to store the weights of the best model
The model will train for 30 epochs but we will use ModelCheckpoint to store the weights of the best performing epoch. We will specify val_acc as the metric to use to define the best model. This means we will keep the weights of the epoch that scores highest in terms of accuracy on the test set.

```
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)
```                               
Now it's time to train the model, here we include the callback to our checkpointer

```
history = classifier.fit_generator(training_set,
                                   steps_per_epoch = 100,
                                   epochs = 20,
                                   callbacks=[checkpointer],
                                   validation_data = test_set,
                                   validation_steps = 50)
```                                 

The model trained for 20 epochs but reached it's best performance at epoch 10. You will notice the message that says:  Epoch 00010: val_acc improved from 0.93333 to 0.95556, saving model to best_weights.hdf5

That means we have now an hdf5 file which stores the weights of that specific epoch, where the accuracy over the test set was of 95,6%

## Load our classifier with the weights of the best model
Now we can load those weights as our final model:
```
classifier.load_weights('best_weights.hdf5')
```

## Saving the complete model
And let's save the final model for usage later:
```
classifier.save('shapes_cnn.h5')
```

## Displaying curves of loss and accuracy during training
Let's now inspect how our model performed over the 30 epochs:

```
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
​
epochs = range(1, len(acc) + 1)
​
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
​
plt.figure()
​
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
​
plt.show()
```

## Classes
Let's clarify now the class number assigned to each of our figures set, since that is how the model will produce it's predictions:

circles: 0
squres: 1
triangles: 2

## Predicting new images
With our model trained and stored, we can load a simple unseen image from our test set and see how it is classified:

```
img_path = 'test_set/triangles/drawing(2).png'
​
img = image.load_img(img_path, target_size=(28, 28))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
​
plt.imshow(img_tensor[0])
plt.show()
​
print(img_tensor.shape)
```

(1, 28, 28, 3)

# predicting images
```
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
​
images = np.vstack([x])
classes = classifier.predict_classes(images, batch_size=10)
print("Predicted class is:",classes)
```
Predicted class is: [2]
The prediction is class [2] which is a triangle

# Visualizing intermediate activations
Quoting François Chollet in his book "DEEP LEARNING with Python" (and I'll quote him a lot in this section):

Intermediate activations are "useful for understanding how successive convnet layers transform their input, and for getting a first idea of the meaning of individual convnet filters."

"The representations learned by convnets are highly amenable to visualization, in large part because they’re representations of visual concepts. Visualizing intermediate activations consists of displaying the feature maps that are output by various convolution and pooling layers in a network, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image."

Next, we’ll get an input image—a picture of a triangle, not part of the images the network was trained on.

"In order to extract the feature maps we want to look at, we’ll create a Keras model that takes batches of images as input, and outputs the activations of all convolution and pooling layers. To do this, we’ll use the Keras class Model. A model is instantiated using two arguments: an input tensor (or list of input tensors) and an output tensor (or list of output tensors). The resulting class is a Keras model, just like the Sequential models, mapping the specified inputs to the specified outputs. What sets the Model class apart is that it allows for models with multiple outputs, unlike Sequential."

## Instantiating a model from an input tensor and a list of output tensors
```
layer_outputs = [layer.output for layer in classifier.layers[:12]] # Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
```
When fed an image input, this model returns the values of the layer activations in the original model

## Running the model in predict mode

```
activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation
```
For instance, this is the activation of the first convolution layer for the image input:
```
first_layer_activation = activations[0]
print(first_layer_activation.shape)
```
(1, 28, 28, 32)
It’s a 28 × 28 feature map with 32 channels. Let’s try plotting the fourth channel of the activation of the first layer of the original model

```
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
```

Even before we try to interpret this activation, let's instead plot all the activations of this same image across each layer

## Visualizing every channel in every intermediate activation
```
layer_names = []
for layer in classifier.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16
​
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
```

So here it is! Let's try to interpret what's going on:

* The first layer acts is arguably retaining the full shape of the triangle, although there are several filters that are not activated and are left blank. At that stage, the activations retain almost all of the information present in the initial picture.
* As we go deeper in the layers, the activations become increasingly abstract and less visually interpretable. They begin to encode higher-level concepts such as single borders, corners and angles. Higher presentations carry increasingly less information about the visual contents of the image, and increasingly more information related to the class of the image.
* As mentioned above, the model stucture is overly complex to the point where we can see our last layers actually not activating at all, there's nothing more to learn at that point.
