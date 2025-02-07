# PokémonVAE
Using a **Variational Autoencoder (VAE)** to generate Pokémon-style images involves the following steps:

### **1. Overview of Variational Autoencoders (VAE)**
A **VAE** is a type of generative model that learns a compressed latent space representation of data and can generate new samples that resemble the training data.

For Pokémon-style image generation, the VAE will:
1. **Encode** Pokémon images into a low-dimensional latent space.
2. **Sample** from the latent space.
3. **Decode** back into Pokémon-style images.

---

### **2. Steps to Build a Pokémon VAE**
#### **A. Data Preparation**
- **Dataset**: Use **Pokémon images** (e.g., from the Pokémon dataset on Kaggle).
- **Preprocessing**: Resize all images to a fixed size (e.g., 64x64), normalize pixel values, and augment the dataset if needed.

#### **B. Model Architecture**
A **VAE** consists of:
1. **Encoder**: Compresses input images into a latent space.
2. **Latent Space**: Learns a probability distribution.
3. **Decoder**: Reconstructs images from latent vectors.

---

### **3. Code Implementation (PyTorch)**
Below is a **VAE implementation** trained on Pokémon images.

#### **Install Dependencies**
```bash
pip install torch torchvision matplotlib numpy
```

#### **Define the VAE Model**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os

# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder_fc(z).view(-1, 128, 8, 8)
        return self.decoder(x), mu, logvar
```

---

#### **4. Training the VAE**
```python
# Load Pokémon dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageFolder(root="./pokemon_images/", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training function
def train_vae(model, dataloader, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.cuda()
            recon, mu, logvar = model(imgs)

            # VAE Loss = Reconstruction Loss + KL Divergence
            recon_loss = criterion(recon, imgs)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / imgs.shape[0]
            loss = recon_loss + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}")

        # Save sample Pokémon images per epoch
        with torch.no_grad():
            sample = torch.randn(16, model.latent_dim).cuda()
            generated_images = model.decoder_fc(sample).view(-1, 128, 8, 8)
            generated_images = model.decoder(generated_images)
            save_image(generated_images, f"generated_pokemon_epoch_{epoch+1}.png", normalize=True)

# Train model
model = VAE().cuda()
train_vae(model, dataloader)
```

---

#### **5. Generating Pokémon-Style Images**
Once trained, generate new Pokémon-style images by sampling the latent space:
```python
def generate_pokemon(model, num_images=16):
    model.eval()
    with torch.no_grad():
        sample = torch.randn(num_images, model.latent_dim).cuda()
        generated_images = model.decoder_fc(sample).view(-1, 128, 8, 8)
        generated_images = model.decoder(generated_images)
        save_image(generated_images, "generated_pokemon.png", normalize=True)
        print("Generated Pokémon images saved!")

generate_pokemon(model)
```

---

### **6. Next Steps & Enhancements**
- **Increase Model Complexity**: Use deeper networks or **GANs** (like Pokémon GANs).
- **Train on a Larger Dataset**: More Pokémon images lead to better diversity.
- **Experiment with Latent Space Interpolation**: Generate in-between Pokémon designs.
- **Use Conditional VAE (cVAE)**: Generate Pokémon by type (e.g., Fire, Water).
- **Combine with Style Transfer**: Add texture patterns from real Pokémon images.

Would you like assistance in fine-tuning or improving the model for higher quality Pokémon generation? 🚀
