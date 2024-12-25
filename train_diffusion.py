import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.blocks import UNet
from data_preparation import generate_snake_data

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
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    unet = UNet(
        cond_channels=COND_CHANNELS,
        depths=DEPTHS,
        channels=CHANNELS,
        attn_depths=ATTN_DEPTHS
    ).to(device)
    action_embedding = ActionEmbedding().to(device)
    
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
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(action_embedding.parameters()),
        lr=LEARNING_RATE
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders()
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
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
        print(
            f"Epoch {epoch}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
