import torch
import numpy as np
from blue import GameNGenSnake, Direction
from collections import deque
from typing import Tuple, List
import matplotlib.pyplot as plt
import imageio


def encode_frame(game: GameNGenSnake) -> torch.Tensor:
    """Convert current game state to a 20x20 tensor.
    
    Returns:
        torch.Tensor: Shape (20, 20) with:
            0 = empty cell
            1 = snake body
            2 = food
    """
    grid = np.zeros((game.state.height, game.state.width), dtype=np.float32)
    
    # Add snake body
    for pos in game.state.snake_positions:
        grid[pos[1], pos[0]] = 1.0
        
    # Add food
    food_pos = game.state.food_position
    grid[food_pos[1], food_pos[0]] = 2.0
    
    return torch.from_numpy(grid)


def generate_snake_data(
    num_episodes: int = 1000,
    max_steps: int = 100,
    history_length: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate training data from snake game episodes.
    
    Args:
        num_episodes: Number of game episodes to simulate
        max_steps: Maximum steps per episode
        history_length: Number of past frames to include
    
    Returns:
        Tuple containing:
        - dataset_tensors: Shape (num_samples, history_length, 20, 20)
          containing frame histories
        - command_tensors: Shape (num_samples,) containing commands
        - next_frame_tensors: Shape (num_samples, 20, 20) containing frames
    """
    # Lists to store collected data
    frame_histories: List[torch.Tensor] = []
    commands: List[int] = []
    next_frames: List[torch.Tensor] = []
    
    # Direction to integer mapping
    dir_to_int = {
        Direction.UP: 0,
        Direction.DOWN: 1,
        Direction.LEFT: 2,
        Direction.RIGHT: 3
    }
    
    for episode in range(num_episodes):
        game = GameNGenSnake()
        frame_history = deque(maxlen=history_length)
        
        # Initialize frame history with initial state
        initial_frame = encode_frame(game)
        for _ in range(history_length):
            frame_history.append(initial_frame)
            
        for step in range(max_steps):
            # Get current frame history
            current_history = torch.stack(list(frame_history))
            
            # Choose a random action
            action = np.random.choice(list(Direction))
            command_int = dir_to_int[action]
            
            # Store current state and action
            frame_histories.append(current_history)
            commands.append(command_int)
            
            # Take action and get next state
            success, _ = game.update(action)
            next_frame = encode_frame(game)
            next_frames.append(next_frame)
            
            # Update frame history
            frame_history.append(next_frame)
            
            # Stop if game is over
            if not success or game.state.game_over:
                break
    
    # Convert lists to tensors
    dataset_tensors = torch.stack(frame_histories)
    command_tensors = torch.tensor(commands, dtype=torch.long)
    next_frame_tensors = torch.stack(next_frames)
    
    return dataset_tensors, command_tensors, next_frame_tensors


def create_game_animation(
    num_episodes=3,
    max_steps=50,
    save_path='game_animation.gif'
):
    """Create an animation of multiple game episodes."""
    frames = []
    fig, ax = plt.subplots(figsize=(8, 8))
    
    dir_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    for episode in range(num_episodes):
        game = GameNGenSnake()
        
        for step in range(max_steps):
            # Clear previous frame
            ax.clear()
            
            # Get current state
            current_frame = encode_frame(game)
            
            # Plot the frame
            im = ax.imshow(current_frame, cmap='RdBu', vmin=0, vmax=2)
            ax.grid(True)
            ax.set_title(
                f'Episode {episode + 1}, Step {step + 1}\n'
                f'Score: {len(game.state.snake_positions)}'
            )
            
            # Add colorbar on first frame
            if episode == 0 and step == 0:
                plt.colorbar(
                    im,
                    ax=ax,
                    label='Cell Type (0=Empty, 1=Snake, 2=Food)'
                )
            
            # Choose random action
            action = np.random.choice(list(Direction))
            command_int = {
                Direction.UP: 0,
                Direction.DOWN: 1,
                Direction.LEFT: 2,
                Direction.RIGHT: 3
            }[action]
            
            # Add action text
            ax.text(
                0.02, 1.05,
                f'Action: {dir_names[command_int]}',
                transform=ax.transAxes,
                fontsize=10
            )
            
            # Save frame
            fig.canvas.draw()
            buf = fig.canvas.tostring_argb()
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                fig.canvas.get_width_height()[::-1] + (4,)
            )
            # Convert ARGB to RGB
            frame = frame[:, :, [1, 2, 3]]
            frames.append(frame)
            
            # Update game state
            success, _ = game.update(action)
            if not success or game.state.game_over:
                break
    
    # Save animation
    imageio.mimsave(save_path, frames, fps=5)
    plt.close()
    return save_path


if __name__ == "__main__":
    # Test data generation
    dataset_tensors, command_tensors, next_frame_tensors = generate_snake_data(
        num_episodes=2,
        max_steps=10,
        history_length=3
    )
    
    print(f"Dataset shape: {dataset_tensors.shape}")
    print(f"Commands shape: {command_tensors.shape}")
    print(f"Next frames shape: {next_frame_tensors.shape}")
    
    # Create and save animation
    animation_path = create_game_animation(num_episodes=3, max_steps=50)
    print(f"Animation saved to: {animation_path}")
