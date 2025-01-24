Here are practical examples of **autoencoders** in Python using TensorFlow/Keras, covering different architectures and applications:

---

### **1. Basic Autoencoder (MNIST Dense)**
```python
import tensorflow as tf
from tensorflow.keras import layers

# Encoder
input_img = layers.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Load MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255

# Train
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                validation_data=(x_test, x_test))
```

---

### **2. Convolutional Autoencoder (Image Denoising)**
```python
# Add noise to images
def add_noise(x, noise_factor=0.5):
    x_noisy = x + noise_factor * tf.random.normal(shape=x.shape)
    return tf.clip_by_value(x_noisy, 0., 1.)

# Build model
input_img = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(x)

# Decoder
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train with noisy data
x_train_noisy = add_noise(x_train.reshape(-1,28,28,1))
x_test_noisy = add_noise(x_test.reshape(-1,28,28,1))

autoencoder.fit(x_train_noisy, x_train.reshape(-1,28,28,1),
                epochs=30,
                batch_size=128,
                validation_data=(x_test_noisy, x_test.reshape(-1,28,28,1)))
```

---

### **3. Variational Autoencoder (VAE)**
```python
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(2 * latent_dim),  # μ and log(σ²)
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28,28))
        ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=False)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        return tf.sigmoid(logits) if apply_sigmoid else logits

# Loss function
@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1,2])
    kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    return -tf.reduce_mean(logpx_z + kl_div)

# Training
vae = VAE(latent_dim=2)
optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        loss = compute_loss(vae, x)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss
```

---

### **4. LSTM Autoencoder (Sequence Data)**
```python
# Generate synthetic sequence data
import numpy as np
data = np.array([np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100) for _ in range(1000)])
data = data.reshape((1000, 100, 1))

# Model
model = tf.keras.Sequential([
    layers.LSTM(64, activation='relu', input_shape=(100,1), return_sequences=True),
    layers.LSTM(32, activation='relu', return_sequences=False),
    layers.RepeatVector(100),
    layers.LSTM(32, activation='relu', return_sequences=True),
    layers.LSTM(64, activation='relu', return_sequences=True),
    layers.TimeDistributed(layers.Dense(1))
])

model.compile(optimizer='adam', loss='mse')
model.fit(data, data, epochs=50, batch_size=32)
```

---

### **5. Autoencoder for Anomaly Detection (ECG)**
```python
# Load ECG dataset
from tensorflow.keras.datasets import ecg
(x_train, _), (x_test, _) = ecg.load_data()

# Simple autoencoder
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(140,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(140, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train, epochs=30, validation_split=0.2)

# Detect anomalies
reconstructions = model.predict(x_test)
mse = np.mean(np.power(x_test - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold
```

---

### **Key Applications**
1. **Dimensionality Reduction**  
2. **Image Denoising**  
3. **Anomaly Detection**  
4. **Feature Learning**  
5. **Data Generation** (VAEs)  
6. **Sequence-to-Sequence Learning**  

### **When to Use Autoencoders**
- Unlabeled data available  
- Need compressed representations  
- Data reconstruction tasks  
- Pre-training for supervised tasks  

### **Performance Tips**
1. Use convolutional layers for image data  
2. Add regularization (Dropout, L1/L2) to prevent overfitting  
3. Use batch normalization between layers  
4. Try different loss functions (MSE, SSIM)  
5. Monitor reconstruction error during training  

Let me know if you'd like to explore any specific architecture or application!
