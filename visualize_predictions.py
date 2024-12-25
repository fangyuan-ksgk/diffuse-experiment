import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def plot_frame_comparison(history_frames, action, true_next_frame, pred_next_frame, save_path=None):
    """Plot history frames, true next frame, and predicted next frame side by side."""
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    
    # Custom colormap for snake game
    colors = ['white', 'blue', 'red']  # 0=empty, 1=snake, 2=food
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Plot history frames
    for i, frame in enumerate(history_frames):
        im = axes[i].imshow(frame, cmap=cmap, vmin=0, vmax=2)
        axes[i].set_title(f'History Frame {i+1}')
        axes[i].axis('off')
        
        # Add grid for better visibility
        axes[i].grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot true next frame
    axes[4].imshow(true_next_frame, cmap=cmap, vmin=0, vmax=2)
    axes[4].set_title('True Next Frame')
    axes[4].axis('off')
    axes[4].grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot predicted next frame
    axes[5].imshow(pred_next_frame, cmap=cmap, vmin=0, vmax=2)
    axes[5].set_title('Predicted Next Frame')
    axes[5].axis('off')
    axes[5].grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Empty', 'Snake', 'Food'])
    
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
    try:
        predictions = np.load(predictions_file, allow_pickle=True)
    except FileNotFoundError: 
        print(f"Error: Could not find predictions file '{predictions_file}'")
        return
    except Exception as e:
        print(f"Error loading predictions: {str(e)}")
        return
    
    # Create directory for individual frame comparisons
    import os
    from tqdm import tqdm
    os.makedirs('prediction_frames', exist_ok=True)
    
    # Plot individual frame comparisons with progress bar
    print("\nGenerating frame comparisons...")
    for i, pred in tqdm(enumerate(predictions), total=len(predictions), desc="Generating frames"):
        try:
            plot_frame_comparison(
                pred['history'],
                pred['action'],
                pred['true_frame'] if 'true_frame' in pred else pred['history'][-1],
                pred['prediction'],
                save_path=f'prediction_frames/frame_{i:03d}.png'
            )
        except Exception as e:
            print(f"\nError processing frame {i}: {str(e)}")
            continue
    
    # Create animation with improved colors
    print("\nCreating animation...")
    try:
        create_prediction_animation(predictions)
        print("\nVisualization complete! Check prediction_frames/ for individual frames")
        print("and prediction_animation.gif for the animation.")
    except Exception as e:
        print(f"\nError creating animation: {str(e)}")

if __name__ == '__main__':
    visualize_prediction_sequence()
