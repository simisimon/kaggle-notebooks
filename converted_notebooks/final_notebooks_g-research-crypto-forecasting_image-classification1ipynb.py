#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

print(f'Tensorflow version: {tf.__version__}')


# In[ ]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])


# In[ ]:


predictions = model(x_train[:1]).numpy()
predictions


# In[ ]:


tf.nn.softmax(predictions).numpy()


# In[ ]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# In[ ]:


loss_fn(y_train[:1], predictions).numpy()


# In[ ]:


model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']
)


# In[ ]:


model.fit(x_train, y_train, epochs=5)


# In[ ]:


model.evaluate(x_test, y_test, verbose=2)


# In[ ]:


probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])


# In[ ]:


probability_model(x_test[:5])

