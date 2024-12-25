import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def plot_frame_comparison(history_frames, action, true_next_frame, pred_next_frame, save_path=None):
    """Plot history frames, true next frame, and predicted next frame side by side."""
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    
    # Plot history frames
    for i, frame in enumerate(history_frames):
        axes[i].imshow(frame, cmap='coolwarm', vmin=0, vmax=2)
        axes[i].set_title(f'History Frame {i+1}')
        axes[i].axis('off')
    
    # Plot true next frame
    axes[4].imshow(true_next_frame, cmap='coolwarm', vmin=0, vmax=2)
    axes[4].set_title('True Next Frame')
    axes[4].axis('off')
    
    # Plot predicted next frame
    axes[5].imshow(pred_next_frame, cmap='coolwarm', vmin=0, vmax=2)
    axes[5].set_title('Predicted Next Frame')
    axes[5].axis('off')
    
    # Add action information
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    plt.suptitle(f'Action: {action_names[action]}', y=1.05)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def create_prediction_animation(predictions, save_path='prediction_animation.gif'):
    """Create an animation comparing true vs predicted frames over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    def update(frame_idx):
        pred = predictions[frame_idx]
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Plot true frame
        ax1.imshow(pred['true_frame'], cmap='coolwarm', vmin=0, vmax=2)
        ax1.set_title('True Frame')
        ax1.axis('off')
        
        # Plot predicted frame
        ax2.imshow(pred['predicted_frame'], cmap='coolwarm', vmin=0, vmax=2)
        ax2.set_title('Predicted Frame')
        ax2.axis('off')
        
        # Add action information
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        plt.suptitle(f'Frame {frame_idx + 1}, Action: {action_names[pred["action"]]}')
    
    anim = FuncAnimation(
        fig, update,
        frames=len(predictions),
        interval=200  # 200ms between frames
    )
    
    anim.save(save_path, writer='pillow')
    plt.close()

def visualize_prediction_sequence(predictions_file='predictions.npy'):
    """Load and visualize a sequence of predictions."""
    predictions = np.load(predictions_file, allow_pickle=True)
    
    # Create directory for individual frame comparisons
    import os
    os.makedirs('prediction_frames', exist_ok=True)
    
    # Plot individual frame comparisons
    for i, pred in enumerate(predictions):
        plot_frame_comparison(
            pred['history'],
            pred['action'],
            pred['true_frame'] if 'true_frame' in pred else pred['history'][-1],
            pred['prediction'],
            save_path=f'prediction_frames/frame_{i:03d}.png'
        )
    
    # Create animation
    create_prediction_animation(predictions)

if __name__ == '__main__':
    visualize_prediction_sequence()
