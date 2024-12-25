import torch
from data_preparation import generate_snake_data
from train_diffusion import HISTORY_LENGTH, COND_CHANNELS, DEPTHS, CHANNELS, ATTN_DEPTHS, UNet, ActionEmbedding

def main():
    # Generate sample data
    frames, commands, next_frames = generate_snake_data(num_episodes=2, max_steps=10, history_length=HISTORY_LENGTH)

    print('\nData shapes:')
    print(f'frames: {frames.shape}')
    print(f'commands: {commands.shape}')
    print(f'next_frames: {next_frames.shape}')

    # Initialize models
    unet = UNet(cond_channels=COND_CHANNELS, depths=DEPTHS, channels=CHANNELS, attn_depths=ATTN_DEPTHS, in_channels=1)
    action_embedding = ActionEmbedding()

    # Get sample batch
    batch_frames = frames[:2]  # Take 2 samples
    batch_commands = commands[:2]
    batch_next_frames = next_frames[:2]

    # Get action embeddings
    action_emb = action_embedding(batch_commands)

    print('\nProcessed shapes:')
    print(f'batch_frames: {batch_frames.shape}')
    print(f'action_emb: {action_emb.shape}')
    print(f'batch_next_frames: {batch_next_frames.shape}')

    # Add noise (simulate training)
    noise = torch.randn_like(batch_next_frames)
    noisy_frames = batch_next_frames + 0.1 * noise  # Shape: [batch_size, 1, height, width]

    print('\nModel input shapes:')
    print(f'noisy_frames: {noisy_frames.shape}')
    print(f'action_emb: {action_emb.shape}')

    # Try forward pass
    try:
        print('\nStarting forward pass...')
        print(f'Input shapes - noisy_frames: {noisy_frames.shape}, action_emb: {action_emb.shape}')
        
        pred_noise, d_outputs, u_outputs = unet(noisy_frames, action_emb)
        
        print('\nSuccess!')
        print(f'Output shapes:')
        print(f'pred_noise: {pred_noise.shape}')
        print(f'num downsampling outputs: {len(d_outputs)}')
        print(f'num upsampling outputs: {len(u_outputs)}')
    except Exception as e:
        print('\nError:', str(e))
        import traceback
        print('\nFull error traceback:')
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
