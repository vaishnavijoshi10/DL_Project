#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


# In[2]:


import os
os.getcwd()


# In[7]:


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS =3 
EPOCHS = 50


# In[8]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle = True,
    image_size = (256,256),
    batch_size = 32
)


# In[ ]:





# In[9]:


class_names = dataset.class_names
class_names


# In[10]:


len(dataset) #68 batches of 32 images


# In[11]:


plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
            plt.subplot(3,4,i+1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[label_batch[i]])
            plt.axis("off")
            print(image_batch.shape)
            print(label_batch.numpy)


# In[12]:


for image_batch, label_batch in dataset.take(1):
    print(image_batch[0].numpy())


# In[13]:


train_size = 0.8
len(dataset)*train_size


# In[14]:


train_ds = dataset.take(54) #arr[:54](till 54 or first 54)
len(train_ds)


# In[15]:


test_ds = dataset.skip(54) #arr[54:](slicing operator(54 onwards))
len(test_ds)


# In[16]:


val_size = 0.1
len(dataset)*val_size


# In[17]:


val_ds = test_ds.take(6)
len(val_ds)


# In[18]:


test_ds = test_ds.skip(6)
len(test_ds)


# In[19]:


def partition_data(ds,train_split = 0.8,val_split = 0.1,test_split = 0.1,shuffle_size = 10000):
    ds_size = len(ds)
    
    train_size = int(train_split*ds_size)
    val_size = int(val_split*ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds,val_ds,test_ds


# In[20]:


train_ds,val_ds,test_ds = partition_data(dataset)


# In[21]:


len(train_ds)


# In[22]:


len(test_ds)


# In[23]:


len(val_ds)


# In[24]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE) #optimizing datasets for fast training
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)


# In[25]:


resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# In[26]:


data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])


# In[27]:


input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes = 12
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation = 'relu',input_shape = input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
model.build(input_shape = input_shape)


# In[28]:


model.summary()


# In[29]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[30]:


history = model.fit(
    train_ds,
    epochs = EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
)


# In[31]:


scores = model.evaluate(test_ds)


# In[32]:


scores


# In[33]:


history


# In[34]:


history.params


# In[35]:


history.history.keys()


# In[36]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[37]:


EPOCHS = 50
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[38]:


import numpy as np


for image_batch, label_batch in test_ds.take(1):
    first_image = image_batch[0].numpy().astype("uint8")
    first_label = int(label_batch[0])
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(image_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])
    
    break


# In[39]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[40]:


import numpy as np
plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[(labels[i])] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")
    break


# In[41]:


#save_model(model, 'model.h5')
#model.save("C:\\Users\\Dell\\Desktop\\DL PROJECT\models\{model_version}")


# In[42]:


model.save('C:/Users/Dell/Desktop/Model/model.h5')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




