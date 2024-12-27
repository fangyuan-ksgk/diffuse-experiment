import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from src.models.blocks import UNet
from data_preparation import generate_snake_data
from scipy import linalg
import numpy as np

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
HISTORY_LENGTH = 4  # Number of past frames to use
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

# Diffusion hyperparameters
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

# Model hyperparameters
COND_CHANNELS = 16  # Dimension of action embedding
DEPTHS = [2, 2, 2]  # Number of ResBlocks at each resolution
CHANNELS = [64, 128, 256]  # Channel counts at each resolution
ATTN_DEPTHS = [False, True, True]  # Apply attention at deeper levels


class DiffusionScheduler:
    """Manages the noise schedule for the diffusion process."""
    def __init__(
        self,
        num_timesteps=NUM_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END
    ):
        self.num_timesteps = num_timesteps
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Pre-compute diffusion parameters
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.]),
            self.alphas_cumprod[:-1]
        ])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        sqrt_term = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = sqrt_term
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        numer = self.betas * (1 - self.alphas_cumprod_prev)
        denom = 1 - self.alphas_cumprod
        self.posterior_variance = numer / denom
    
    def add_noise(self, x_start, t, noise=None):
        """Add noise to the input according to the noise schedule."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Get noise schedule parameters for timestep t
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Apply noise transformation
        noisy_x = alpha_t * x_start + sigma_t * noise
        return noisy_x, noise
    
    def get_loss_weight(self, t):
        """Get loss weighting for timestep t."""
        alpha_t = self.sqrt_alphas_cumprod[t]
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t]
        return 1 / (alpha_t * sigma_t)


class ActionEmbedding(nn.Module):
    """Embeds discrete actions into continuous vectors."""
    def __init__(self, num_actions=4, embedding_dim=COND_CHANNELS):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)
    
    def forward(self, action_indices):
        return self.embedding(action_indices)


def compute_fda(real_features, generated_features):
    """Compute Fréchet Distance between real and generated feature distributions."""
    # Calculate mean and covariance
    mu1 = real_features.mean(dim=0)
    sigma1 = torch.cov(real_features.T)
    mu2 = generated_features.mean(dim=0)
    sigma2 = torch.cov(generated_features.T)
    
    # Convert to numpy for scipy's implementation
    mu1, mu2 = mu1.cpu().numpy(), mu2.cpu().numpy()
    sigma1, sigma2 = sigma1.cpu().numpy(), sigma2.cpu().numpy()
    
    # Calculate squared difference between means
    diff = mu1 - mu2
    
    # Calculate matrix sqrt of the product of covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check for numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate Fréchet Distance
    fda = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fda)


def create_dataloaders(num_episodes=1000, max_steps=100):
    """Create training and validation dataloaders."""
    # Generate data
    frames, commands, next_frames = generate_snake_data(
        num_episodes=num_episodes,
        max_steps=max_steps,
        history_length=HISTORY_LENGTH
    )
    
    # Split into train/val (90/10)
    split_idx = int(0.9 * len(frames))
    train_data = TensorDataset(
        frames[:split_idx],
        commands[:split_idx],
        next_frames[:split_idx]
    )
    val_data = TensorDataset(
        frames[split_idx:],
        commands[split_idx:],
        next_frames[split_idx:]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE
    )
    
    return train_loader, val_loader


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    model_config = {
        'cond_channels': COND_CHANNELS,
        'depths': DEPTHS,
        'channels': CHANNELS,
        'attn_depths': ATTN_DEPTHS
    }
    unet = UNet(**model_config).to(device)
    action_embedding = ActionEmbedding().to(device)
    
    # Initialize feature extractor for FDA
    feature_extractor = resnet18(pretrained=True).to(device)
    # Remove classification layer
    feature_extractor.fc = nn.Identity()
    feature_extractor.eval()
    # Freeze feature extractor parameters
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    # Initialize diffusion scheduler
    scheduler = DiffusionScheduler()
    
    # Move scheduler parameters to device
    scheduler_params = [
        'betas', 'alphas', 'alphas_cumprod',
        'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod'
    ]
    for param in scheduler_params:
        setattr(scheduler, param, getattr(scheduler, param).to(device))
    
    # Initialize optimizer
    # Combine model parameters for optimization
    model_params = list(unet.parameters()) + list(action_embedding.parameters())
    optimizer = torch.optim.AdamW(model_params, lr=LEARNING_RATE)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume)
            unet.load_state_dict(checkpoint['model_state_dict'])
            action_embedding.load_state_dict(checkpoint['action_embedding_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders()
    
    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Training
        unet.train()
        train_loss = 0.0
        for batch_idx, batch_data in enumerate(train_loader):
            frames, actions, next_frames = batch_data
            frames = frames.to(device)
            actions = actions.to(device)
            next_frames = next_frames.to(device)
            
            # Sample timesteps uniformly for each item in batch
            batch_size = frames.shape[0]
            t = torch.randint(
                0, NUM_TIMESTEPS,
                (batch_size,),
                device=device
            )
            
            # Get action embeddings
            action_emb = action_embedding(actions)
            
            # Add noise to target frames
            noisy_frames, target_noise = scheduler.add_noise(next_frames, t)
            
            # Predict noise
            pred_noise, _, _ = unet(noisy_frames, action_emb)
            
            # Compute loss with importance sampling
            loss_weights = scheduler.get_loss_weight(t)
            squared_diff = torch.square(target_noise - pred_noise)
            loss = torch.mean(loss_weights * squared_diff)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}, "
                    f"Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        train_loss /= len(train_loader)
        
        # Validation
        unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, actions, next_frames in val_loader:
                frames = frames.to(device)
                actions = actions.to(device)
                next_frames = next_frames.to(device)
                
                # Sample timesteps uniformly
                batch_size = frames.shape[0]
                t = torch.randint(
                    0, NUM_TIMESTEPS,
                    (batch_size,),
                    device=device
                )
                
                # Get action embeddings
                action_emb = action_embedding(actions)
                
                # Add noise to target frames
                noisy_frames, target_noise = scheduler.add_noise(
                    next_frames, t
                )
                
                # Predict noise
                pred_noise, _, _ = unet(noisy_frames, action_emb)
                
                # Compute loss
                loss_weights = scheduler.get_loss_weight(t)
                squared_diff = torch.square(target_noise - pred_noise)
                batch_loss = torch.mean(loss_weights * squared_diff)
                val_loss += batch_loss.item()
        
        val_loss /= len(val_loader)
        
        # Compute FDA score on validation set
        real_features = []
        generated_features = []
        with torch.no_grad():
            for frames, actions, next_frames in val_loader:
                frames = frames.to(device)
                actions = actions.to(device)
                next_frames = next_frames.to(device)
                
                # Get action embeddings
                action_emb = action_embedding(actions)
                
                # Generate samples
                noisy = torch.randn_like(next_frames)
                for t in reversed(range(NUM_TIMESTEPS)):
                    t_batch = torch.full((frames.shape[0],), t, device=device)
                    pred_noise, _, _ = unet(noisy, action_emb, t_batch)
                    alpha_t = scheduler.sqrt_alphas_cumprod[t]
                    sigma_t = scheduler.sqrt_one_minus_alphas_cumprod[t]
                    noisy = (noisy - sigma_t * pred_noise) / alpha_t
                    if t > 0:
                        noise = torch.randn_like(noisy)
                        sigma = scheduler.posterior_variance[t].sqrt()
                        noisy = noisy + sigma * noise
                generated = noisy.clamp(-1, 1)
                
                # Extract features
                real_batch = feature_extractor(next_frames)
                gen_batch = feature_extractor(generated)
                real_features.append(real_batch)
                generated_features.append(gen_batch)
        
        # Compute FDA score
        real_features = torch.cat(real_features, dim=0)
        generated_features = torch.cat(generated_features, dim=0)
        fda_score = compute_fda(real_features, generated_features)
        
        print(
            f"Epoch {epoch}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"FDA Score: {fda_score:.4f}"
        )
        
        # Save checkpoint
        checkpoint_name = f'checkpoint_epoch_{epoch:03d}.pt'
        checkpoint_path = os.path.join('checkpoints', checkpoint_name)
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'action_embedding_state_dict': action_embedding.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
