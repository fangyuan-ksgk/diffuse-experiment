import torch
import numpy as np
from src.models.blocks import UNet
from train_diffusion import DiffusionScheduler
from data_preparation import GameNGenSnake, Direction, encode_frame

def load_trained_model():
    model = UNet(
        cond_channels=16,  # Action embedding dimension
        depths=[2, 2, 2],  # Number of ResBlocks at each resolution
        channels=[32, 64, 128],  # Channel counts at each resolution
        attn_depths=[False, True, True],  # Apply attention at deeper levels
        in_channels=1  # Single channel for game state
    )
    model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
    model.eval()
    return model

def generate_predictions(model, num_samples=5, sequence_length=10):
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
                current_frame = torch.tensor(history[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 20, 20]
                a = torch.tensor([action], dtype=torch.long).unsqueeze(0)
                
                # Generate prediction
                with torch.no_grad():
                    # Start from random noise
                    x_t = torch.randn_like(current_frame)  # [1, 1, 20, 20]
                    
                    # Denoise gradually
                    for t in reversed(range(0, 1000, 100)):  # Sample fewer timesteps for speed
                        t_tensor = torch.tensor([t], dtype=torch.long)
                        noise_scale = scheduler.get_noise_scale(t_tensor)
                        
                        # Prepare action embedding
                        action_emb = torch.zeros(1, 16)  # cond_channels=16
                        action_emb[0, a.item()] = 1.0  # One-hot encoding
                        
                        # Model prediction
                        pred, _, _ = model(x_t, action_emb)
                        
                        # Update sample
                        if t > 0:
                            noise = torch.randn_like(x_t)
                            t_next = t - 100
                            next_noise_scale = scheduler.get_noise_scale(torch.tensor([t_next]))
                            x_t = (
                                pred * (1 - next_noise_scale) +
                                noise * next_noise_scale
                            )
                        else:
                            x_t = pred
                
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
    print("Loading model...")
    model = load_trained_model()
    
    print("Generating predictions...")
    predictions = generate_predictions(model)
    
    print("Saving predictions...")
    np.save('predictions.npy', predictions)
    print("Done! Generated", len(predictions), "predictions.")
