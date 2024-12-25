import torch
import numpy as np
from src.models.blocks import UNet
from train_diffusion import DiffusionScheduler
from data_preparation import GameNGenSnake

def load_trained_model():
    model = UNet(
        in_channels=5,  # 4 history frames + 1 action channel
        out_channels=1,  # Next frame prediction
        channels=[32, 64, 128],
        depths=[2, 2, 2]
    )
    model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
    model.eval()
    return model

def generate_predictions(model, num_samples=5, sequence_length=10):
    scheduler = DiffusionScheduler(num_timesteps=1000)
    game = GameNGenSnake(grid_size=20)
    predictions = []
    
    for _ in range(num_samples):
        # Initialize game state
        game.reset()
        history = []
        actions = []
        
        # Generate sequence
        for t in range(sequence_length):
            # Get current state
            state = game.get_state()
            history.append(state)
            if len(history) > 4:
                history = history[-4:]
            
            # Random action for demonstration
            action = np.random.choice([0, 1, 2, 3])  # UP, DOWN, LEFT, RIGHT
            actions.append(action)
            
            # If we have enough history, generate prediction
            if len(history) == 4:
                # Prepare input
                x = torch.tensor(np.stack(history), dtype=torch.float32).unsqueeze(0)
                a = torch.tensor([action], dtype=torch.long).unsqueeze(0)
                
                # Generate prediction
                with torch.no_grad():
                    # Start from random noise
                    x_t = torch.randn(1, 1, 20, 20)
                    
                    # Denoise gradually
                    for t in reversed(range(0, 1000, 100)):  # Sample fewer timesteps for speed
                        t_tensor = torch.tensor([t], dtype=torch.long)
                        noise_scale = scheduler.get_noise_scale(t_tensor)
                        
                        # Model prediction
                        pred = model(x_t, noise_scale, x, a)
                        
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
            game.step(action)
    
    return predictions

if __name__ == '__main__':
    print("Loading model...")
    model = load_trained_model()
    
    print("Generating predictions...")
    predictions = generate_predictions(model)
    
    print("Saving predictions...")
    np.save('predictions.npy', predictions)
    print("Done! Generated", len(predictions), "predictions.")
