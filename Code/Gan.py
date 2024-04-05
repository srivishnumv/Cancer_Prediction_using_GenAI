#!/usr/bin/env python
# coding: utf-8

# Import Necessary Libraries, 

# In[4]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam


# Load the dataset used in breast-cancer.ipynb

# In[5]:


data = pd.read_csv('C:/Users/Srivishnu/OneDrive/Desktop/Project_Code/Cancer_Prediction_using_GenAI/data.csv')


# Data preprocessing and feature selection

# In[6]:


# Separate features (X) and labels (y)
X = data.drop(columns=['diagnosis']).values
y = data['diagnosis'].values

# Normalize features to range [0, 1]
X_normalized = (X - X.min()) / (X.max() - X.min())

# Define the size of the latent space
latent_dim = 100


# Generator function

# In[7]:


def define_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(X_normalized.shape[1], activation='tanh'))
    return model


# Discriminator Function

# In[8]:


def define_discriminator(input_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Use the learning_rate parameter instead of lr
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model


# GAN function

# In[9]:


def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    # Use the learning_rate parameter instead of lr
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model



# Synthetic real data generation

# In[10]:


def generate_real_samples(dataset, n_samples):
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[idx]
    y = np.ones((n_samples, 1))
    return X, y

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# Synthetic Fake data generation

# In[11]:


def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


# GAN Training

# In[12]:


def train_gan(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=500, n_batch=128):
    for epoch in range(n_epochs):
        for batch in range(len(dataset) // n_batch):
            X_real, y_real = generate_real_samples(dataset, n_batch)
            d_loss_real, _ = discriminator.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_batch)
            d_loss_fake, _ = discriminator.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print(f'Epoch {epoch+1}/{n_epochs}, Batch {batch+1}/{len(dataset)//n_batch}, D_real_loss={d_loss_real}, D_fake_loss={d_loss_fake}, G_loss={g_loss}')


# In[13]:


# Define discriminator model
discriminator = define_discriminator(X_normalized.shape[1])

# Define generator model
generator = define_generator(latent_dim)

# Define GAN model
gan_model = define_gan(generator, discriminator)

# Train GAN
train_gan(generator, discriminator, gan_model, X_normalized, latent_dim)


# Data augmentation

# In[14]:


# Generate synthetic samples using GAN
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
import numpy as np
# Generate synthetic samples using GAN
def generate_synthetic_samples(generator, latent_dim, num_samples):
    # Generate random noise as input to the generator
    latent_points = np.random.randn(num_samples, latent_dim)
    # Generate synthetic samples
    X_synthetic = generator.predict(latent_points)
    return X_synthetic

# Number of synthetic samples to generate
num_synthetic_samples = 1000  # Adjust as needed

# Generate synthetic samples
X_synthetic = generate_synthetic_samples(generator, latent_dim, num_synthetic_samples)

# Combine original and synthetic samples
X_augmented = np.vstack((X_normalized, X_synthetic))
y_augmented = np.concatenate((y, np.ones(num_synthetic_samples)))  # Assuming all synthetic samples are labeled as 1


# Augment dataset with synthetic samples



# Augmented dataset preprocessing

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Shuffle the augmented dataset
X_augmented_shuffled, y_augmented_shuffled = shuffle(X_augmented, y_augmented)

# Split the augmented dataset into training and testing sets
X_train_augmented, X_test_augmented, y_train_augmented, y_test_augmented = train_test_split(X_augmented_shuffled, y_augmented_shuffled, test_size=0.2, random_state=42)


# In[16]:


from sklearn.preprocessing import LabelEncoder

# Convert labels to strings to handle mixed data types
y_train_augmented_str = y_train_augmented.astype(str)

# Encode labels for training data
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_augmented_str)

# Print unique encoded labels for training data
print(np.unique(y_train_encoded))

# Convert labels to strings for testing data
y_test_augmented_str = y_test_augmented.astype(str)

# Encode labels for testing data
y_test_encoded = label_encoder.transform(y_test_augmented_str)

# Print unique encoded labels for testing data
print(np.unique(y_test_encoded))

combined_data = pd.concat([data, pd.DataFrame(X_synthetic)], axis=0)
synthetic_labels = np.ones((num_synthetic_samples,))
combined_labels = np.concatenate([y, synthetic_labels])
combined_data.to_csv('combined_data.csv', index=False)
np.savetxt('combined_labels.csv', combined_labels, delimiter=',')





