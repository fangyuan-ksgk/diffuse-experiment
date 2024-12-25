import torch
import numpy as np
from src.models.blocks import UNet
from train_diffusion import DiffusionScheduler, ActionEmbedding
from data_preparation import GameNGenSnake, Direction, encode_frame

def load_trained_models():
    # Load UNet
    model = UNet(
        cond_channels=16,  # Action embedding dimension
        depths=[2, 2, 2],  # Number of ResBlocks at each resolution
        channels=[32, 64, 128],  # Channel counts at each resolution
        attn_depths=[False, True, True],  # Apply attention at deeper levels
        in_channels=1  # Single channel for game state
    )
    model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
    model.eval()
    
    # Load action embedding
    action_embedding = ActionEmbedding()
    action_embedding.load_state_dict(torch.load('action_embedding_weights.pth', map_location='cpu'))
    action_embedding.eval()
    
    return model, action_embedding

def generate_predictions(model, action_embedding, num_samples=5, sequence_length=10):
    scheduler = DiffusionScheduler(num_timesteps=1000)
    dir_to_int = {
        Direction.UP: 0,
        Direction.DOWN: 1,
        Direction.LEFT: 2,
        Direction.RIGHT: 3
    }
    predictions = []
    
    for _ in range(num_samples):
        # Initialize game state
        game = GameNGenSnake()
        history = []
        actions = []
        
        # Generate sequence
        for t in range(sequence_length):
            # Get current state
            current_frame = encode_frame(game)
            history.append(current_frame)
            if len(history) > 4:
                history = history[-4:]
            
            # Random action for demonstration
            action = np.random.choice(list(Direction))
            action_int = dir_to_int[action]
            actions.append(action_int)
            
            # If we have enough history, generate prediction
            if len(history) == 4:
                # Prepare input (use only the most recent frame)
                current_frame = history[-1].clone().detach().unsqueeze(0).unsqueeze(0)  # [1, 1, 20, 20]
                
                # Initialize with random noise
                x_t = torch.randn_like(current_frame)  # [1, 1, 20, 20]
                
                # Generate prediction
                with torch.no_grad():
                    # Start from random noise
                    x_t = torch.randn_like(current_frame)  # [1, 1, 20, 20]
                    
                    # Denoise gradually
                    for t in reversed(range(0, 1000, 100)):  # Sample fewer timesteps for speed
                        t_tensor = torch.tensor([t], dtype=torch.long)
                        
                        # Get action embedding (ensure action is a tensor)
                        action_tensor = torch.tensor([action_int], dtype=torch.long)
                        action_emb = action_embedding(action_tensor)
                        
                        # Predict noise
                        pred_noise, _, _ = model(x_t, action_emb)
                        
                        # Get noise schedule parameters for timestep t
                        alpha_t = scheduler.sqrt_alphas_cumprod[t]
                        sigma_t = scheduler.sqrt_one_minus_alphas_cumprod[t]
                        
                        # Remove predicted noise and scale back
                        x_start = (x_t - sigma_t * pred_noise) / alpha_t
                        
                        # If not the last step, add noise for next timestep
                        if t > 0:
                            t_next = t - 100
                            noise = torch.randn_like(x_t)
                            x_t, _ = scheduler.add_noise(x_start, torch.tensor([t_next]), noise)
                
                predictions.append({
                    'history': history.copy(),
                    'action': action,
                    'prediction': x_t.squeeze().numpy()
                })
            
            # Update game state
            success, _ = game.update(action)
            if not success or game.state.game_over:
                break
    
    return predictions

if __name__ == '__main__':
    print("Loading models...")
    model, action_embedding = load_trained_models()
    
    print("Generating predictions...")
    predictions = generate_predictions(model, action_embedding)
    
    print("Saving predictions...")
    np.save('predictions.npy', predictions)
    print("Done! Generated", len(predictions), "predictions.")
