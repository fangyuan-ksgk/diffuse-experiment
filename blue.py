#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hope to do Autoregressive prediction ... 

import random
import time
from enum import Enum
from typing import List, Tuple, Optional

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class GameState:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.snake_positions = [(width//2, height//2)]
        self.direction = Direction.RIGHT
        self.food_position = self._generate_food()
        self.score = 0
        self.game_over = False
        self.narrative_history = []
        
    def _generate_food(self) -> Tuple[int, int]:
        while True:
            pos = (random.randint(0, self.width-1), 
                  random.randint(0, self.height-1))
            if pos not in self.snake_positions:
                return pos

class NarrativeGenerator:
    def __init__(self):
        self.movement_templates = [
            "The snake slithers {direction}ward, seeking its prey.",
            "Moving {direction}, the serpent explores its domain.",
            "With determination, our hero ventures {direction}.",
        ]
        
        self.food_templates = [
            "The snake discovers a delicious meal! Score: {score}",
            "Victory! Another morsel consumed. Score: {score}",
            "The hunt is successful as the snake grows stronger. Score: {score}",
        ]
        
        self.collision_templates = [
            "Oh no! The snake meets an unfortunate end at {location}.",
            "The journey ends as our hero collides with {location}.",
            "Game Over: A fatal mistake at {location}.",
        ]
        
        self.close_call_templates = [
            "A narrow escape as the snake barely avoids {location}!",
            "Dancing with danger near {location}!",
            "The snake skillfully maneuvers past {location}.",
        ]

    def generate_movement_narrative(self, direction: Direction) -> str:
        template = random.choice(self.movement_templates)
        return template.format(direction=direction.name.lower())
    
    def generate_food_narrative(self, score: int) -> str:
        template = random.choice(self.food_templates)
        return template.format(score=score)
    
    def generate_collision_narrative(self, location: str) -> str:
        template = random.choice(self.collision_templates)
        return template.format(location=location)
    
    def generate_close_call_narrative(self, location: str) -> str:
        template = random.choice(self.close_call_templates)
        return template.format(location=location)

class GameNGenSnake:
    def __init__(self):
        self.state = GameState()
        self.narrator = NarrativeGenerator()
        
    def update(self, direction: Optional[Direction] = None) -> Tuple[bool, str]:
        if self.state.game_over:
            return False, "Game is already over!"
            
        if direction:
            self.state.direction = direction
            
        # Get current head position
        head_x, head_y = self.state.snake_positions[0]
        
        # Calculate new head position
        if self.state.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.state.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.state.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        else:  # RIGHT
            new_head = (head_x + 1, head_y)
            
        # Check for collisions
        if (new_head[0] < 0 or new_head[0] >= self.state.width or
            new_head[1] < 0 or new_head[1] >= self.state.height or
            new_head in self.state.snake_positions):
            self.state.game_over = True
            narrative = self.narrator.generate_collision_narrative(f"position {new_head}")
            return False, narrative
            
        # Move snake
        self.state.snake_positions.insert(0, new_head)
        
        # Check if food is eaten
        narrative = self.narrator.generate_movement_narrative(self.state.direction)
        if new_head == self.state.food_position:
            self.state.score += 1
            self.state.food_position = self._generate_food()
            food_narrative = self.narrator.generate_food_narrative(self.state.score)
            narrative = f"{narrative}\n{food_narrative}"
        else:
            self.state.snake_positions.pop()
            
        # Check for close calls
        head = self.state.snake_positions[0]
        dangerous_positions = [
            (head[0]+1, head[1]), (head[0]-1, head[1]),
            (head[0], head[1]+1), (head[0], head[1]-1)
        ]
        
        for pos in dangerous_positions:
            if (pos in self.state.snake_positions[1:] or
                pos[0] < 0 or pos[0] >= self.state.width or
                pos[1] < 0 or pos[1] >= self.state.height):
                close_call = self.narrator.generate_close_call_narrative(f"position {pos}")
                narrative = f"{narrative}\n{close_call}"
                break
                
        self.state.narrative_history.append(narrative)
        return True, narrative

    def _generate_food(self) -> Tuple[int, int]:
        return self.state.food_position

# Example usage
def main():
    game = GameNGenSnake()
    
    # Sample game loop
    moves = [Direction.RIGHT, Direction.RIGHT, Direction.DOWN, 
             Direction.DOWN, Direction.LEFT, Direction.UP]
    
    for move in moves:
        success, narrative = game.update(move)
        print("\nNarrative:")
        print(narrative)
        print("\nSnake Position:", game.state.snake_positions)
        print("Food Position:", game.state.food_position)
        print("Score:", game.state.score)
        if not success:
            break
        time.sleep(1)

if __name__ == "__main__":
    main()


# In[ ]:




