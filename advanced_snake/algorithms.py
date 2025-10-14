"""
Algorithms module for the Snake Game.
Implements A* and Dijkstra's algorithms for automatic snake movement.
"""

import heapq
from constants import *

class SnakeAlgorithms:
    def __init__(self, game_engine):
        """Initialize algorithms with a reference to the game engine."""
        self.game_engine = game_engine
        
    def get_next_move_astar(self):
        """
        Determine next move using A* algorithm.
        A* uses heuristic (Manhattan distance) to guide search more efficiently.
        """
        # Get current game state
        snake_head = self.game_engine.get_snake_head()
        food = self.game_engine.food
        
        # If no path is found with A*, try a safe move
        path = self._find_path_astar(snake_head, food)
        if not path or len(path) == 0:
            return self._get_safe_move()
        
        # Get the next position from the path
        next_pos = path[0]
        
        # Convert to direction
        return self._get_direction_from_positions(snake_head, next_pos)
    
    def get_next_move_dijkstra(self):
        """
        Determine next move using Dijkstra's algorithm.
        Dijkstra's finds the shortest path without using a heuristic.
        """
        # Get current game state
        snake_head = self.game_engine.get_snake_head()
        food = self.game_engine.food
        
        # Find path using Dijkstra's algorithm
        path = self._find_path_dijkstra(snake_head, food)
        
        # If no path is found, try a safe move
        if not path or len(path) == 0:
            return self._get_safe_move()
        
        # Get the next position from the path
        next_pos = path[0]
        
        # Convert to direction
        return self._get_direction_from_positions(snake_head, next_pos)
    
    def _find_path_astar(self, start, end):
        """
        Find a path from start to end using A* algorithm.
        Uses Manhattan distance as heuristic.
        """
        # Priority queue for open set
        open_set = []
        # Using a counter to break ties consistently
        counter = 0
        
        # Add start position to open set
        # Format: (f_score, counter, position, path)
        # f_score = g_score (distance from start) + h_score (heuristic to end)
        heapq.heappush(open_set, (0, counter, start, []))
        
        # Keep track of visited positions
        closed_set = set()
        
        while open_set:
            # Get position with lowest f_score
            _, _, current, path = heapq.heappop(open_set)
            
            # If we reached the end, return the path
            if current == end:
                return path + [current]
            
            # Skip if already visited
            if current in closed_set:
                continue
            
            # Add to closed set
            closed_set.add(current)
            
            # Get valid moves from current position
            row, col = current
            
            # Generate neighbors
            directions = [UP, DOWN, LEFT, RIGHT]
            for d_row, d_col in directions:
                new_row = row + d_row
                new_col = col + d_col
                
                # Check if the new position is within boundaries
                if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                    continue  # Skip positions outside boundaries
                
                neighbor = (new_row, new_col)
                
                # Skip if neighbor is in the snake's body (except tail if not growing)
                snake_body = self.game_engine.get_snake_body()
                if len(snake_body) > 0 and start != self.game_engine.food:
                    # The tail will move, so we can move into the tail's current position
                    snake_body = snake_body[:-1]
                
                if neighbor in snake_body or neighbor in closed_set:
                    continue
                    
                # Calculate g_score (distance from start)
                g_score = len(path) + 1
                
                # Calculate h_score (Manhattan distance to end)
                h_score = abs(new_row - end[0]) + abs(new_col - end[1])
                
                # Calculate f_score
                f_score = g_score + h_score
                
                # Add to open set
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor, path + [current]))
        
        # No path found
        return None
    
    def _find_path_dijkstra(self, start, end):
        """
        Find a path from start to end using Dijkstra's algorithm.
        """
        # Priority queue for open set
        open_set = []
        # Using a counter to break ties consistently
        counter = 0
        
        # Add start position to open set
        # Format: (distance, counter, position, path)
        heapq.heappush(open_set, (0, counter, start, []))
        
        # Keep track of visited positions
        closed_set = set()
        
        while open_set:
            # Get position with lowest distance
            dist, _, current, path = heapq.heappop(open_set)
            
            # If we reached the end, return the path
            if current == end:
                return path + [current]
            
            # Skip if already visited
            if current in closed_set:
                continue
            
            # Add to closed set
            closed_set.add(current)
            
            # Get valid moves from current position
            row, col = current
            
            # Generate neighbors
            directions = [UP, DOWN, LEFT, RIGHT]
            for d_row, d_col in directions:
                new_row = row + d_row
                new_col = col + d_col
                
                # Check if the new position is within boundaries
                if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                    continue  # Skip positions outside boundaries
                
                neighbor = (new_row, new_col)
                
                # Skip if neighbor is in the snake's body (except tail if not growing)
                snake_body = self.game_engine.get_snake_body()
                if len(snake_body) > 0 and start != self.game_engine.food:
                    # The tail will move, so we can move into the tail's current position
                    snake_body = snake_body[:-1]
                
                if neighbor in snake_body or neighbor in closed_set:
                    continue
                    
                # Calculate new distance
                new_dist = dist + 1
                
                # Add to open set
                counter += 1
                heapq.heappush(open_set, (new_dist, counter, neighbor, path + [current]))
        
        # No path found
        return None
    
    def _get_direction_from_positions(self, current_pos, next_pos):
        """Convert from current position to next position to a direction vector."""
        current_row, current_col = current_pos
        next_row, next_col = next_pos
        
        # Calculate the difference directly (no wrapping with boundaries)
        row_diff = next_row - current_row
        col_diff = next_col - current_col
        
        # Determine direction
        if row_diff == -1:
            return UP
        elif row_diff == 1:
            return DOWN
        elif col_diff == -1:
            return LEFT
        elif col_diff == 1:
            return RIGHT
        
        return self._get_safe_move()  # Fallback
    
    def _get_safe_move(self):
        """Get a safe move if no path is found to the food."""
        valid_moves = self.game_engine.get_valid_moves()
        
        if not valid_moves:
            # No safe moves, return current direction as last resort
            return self.game_engine.direction
        
        # Prefer moves that don't lead to dead ends
        safe_moves = []
        for move in valid_moves:
            head_row, head_col = self.game_engine.get_snake_head()
            new_row = head_row + move[0]
            new_col = head_col + move[1]
            
            # Check if the new position is within boundaries
            if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                continue
                
            new_pos = (new_row, new_col)
            
            # Check if this move leads to a position with at least one valid move
            temp_valid_moves = self._get_valid_moves_from_position(new_pos)
            if len(temp_valid_moves) > 0:
                safe_moves.append(move)
        
        # If there are safe moves, return one of them
        if safe_moves:
            # Prefer moving towards food if possible
            food = self.game_engine.food
            head = self.game_engine.get_snake_head()
            
            # Calculate direction to food
            food_row, food_col = food
            head_row, head_col = head
            
            # Try to move closer to food
            if food_row < head_row and UP in safe_moves:
                return UP
            elif food_row > head_row and DOWN in safe_moves:
                return DOWN
            elif food_col < head_col and LEFT in safe_moves:
                return LEFT
            elif food_col > head_col and RIGHT in safe_moves:
                return RIGHT
            
            # If no direction is closer to food, return first safe move
            return safe_moves[0]
        
        # If no safe moves, return first valid move
        return valid_moves[0]
    
    def _get_valid_moves_from_position(self, position):
        """Get valid moves from a specific position."""
        row, col = position
        possible_moves = [UP, DOWN, LEFT, RIGHT]
        valid_moves = []
        
        for move in possible_moves:
            new_row = row + move[0]
            new_col = col + move[1]
            
            # Check if the new position is within boundaries
            if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                continue
                
            new_pos = (new_row, new_col)
            
            # Check if this position is safe (not occupied by snake except tail)
            snake_body = self.game_engine.get_snake_body()
            if len(snake_body) > 0:
                # The tail will move, so we can move into the tail's current position
                if new_pos != snake_body[-1] or self.game_engine.get_snake_head() == self.game_engine.food:
                    if new_pos in snake_body:
                        continue
            
            valid_moves.append(move)
        
        return valid_moves