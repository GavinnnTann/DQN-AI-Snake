"""
Algorithms module for the Snake Game.
Implements A*, Dijkstra's, and Hamiltonian cycle algorithms for automatic snake movement.
"""

import heapq
from constants import *

class SnakeAlgorithms:
    def __init__(self, game_engine):
        """Initialize algorithms with a reference to the game engine."""
        self.game_engine = game_engine
        self.hamiltonian_cycle = None  # Cache the Hamiltonian cycle
        self.cycle_index = {}  # Maps position to index in cycle
        
        # DHCR path commitment state (AlphaPhoenix approach)
        # Instead of re-evaluating every step, we commit to entire shortcut paths
        self.committed_path = None  # Current shortcut path being executed
        self.committed_path_index = 0  # Current position in committed path
        
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
        if not path or len(path) < 2:
            return self._get_safe_move()
        
        # Get the next position from the path (skip path[0] which is current position)
        next_pos = path[1]
        
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
        if not path or len(path) < 2:
            return self._get_safe_move()
        
        # Get the next position from the path (skip path[0] which is current position)
        next_pos = path[1]
        
        # Convert to direction
        return self._get_direction_from_positions(snake_head, next_pos)
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
                
                # Skip if neighbor is in the snake's body
                snake_body = self.game_engine.get_snake_body()
                
                # The tail will move away when the snake advances
                if len(snake_body) > 1:
                    # Exclude the tail - it will move away
                    snake_body_for_collision = snake_body[:-1]
                else:
                    # Snake is length 1, can't exclude tail
                    snake_body_for_collision = snake_body
                
                if neighbor in snake_body_for_collision or neighbor in closed_set:
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
                
                # Skip if neighbor is in the snake's body
                snake_body = self.game_engine.get_snake_body()
                
                # Important: The tail will move away UNLESS we just ate food and are growing
                # We can determine if the snake will grow by checking if current position has food
                # But in A*, we're exploring from 'start', not necessarily the actual head
                # So we use a simpler rule: tail can be used as valid move (it moves away)
                if len(snake_body) > 1:
                    # Exclude the tail - it will move away when the snake advances
                    snake_body_for_collision = snake_body[:-1]
                else:
                    # Snake is length 1, can't exclude tail
                    snake_body_for_collision = snake_body
                
                if neighbor in snake_body_for_collision or neighbor in closed_set:
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
    
    def get_next_move_hamiltonian(self):
        """
        Determine next move using Hamiltonian cycle algorithm.
        
        This follows a pre-computed Hamiltonian cycle that visits every cell
        in the grid exactly once and returns to the start. This guarantees
        the snake will eventually collect all food and win the game.
        
        Based on AlphaPhoenix's approach: creates a zigzag pattern where
        columns alternate going up and down, forming a closed cycle.
        
        Returns:
            tuple: Direction tuple (UP, DOWN, LEFT, or RIGHT)
        """
        # Build cycle if not already cached
        if self.hamiltonian_cycle is None:
            self._build_hamiltonian_cycle()
        
        head = self.game_engine.get_snake_head()
        
        # Find current position in cycle
        if head not in self.cycle_index:
            # Fallback to A* if snake is somehow off-path (shouldn't happen)
            print(f"[Hamiltonian] Warning: Snake at {head} is off-path, using A*")
            return self.get_next_move_astar()
        
        current_idx = self.cycle_index[head]
        
        # Get next position in cycle
        next_idx = (current_idx + 1) % len(self.hamiltonian_cycle)
        next_pos = self.hamiltonian_cycle[next_idx]
        
        # Calculate direction to next position
        direction = self._get_direction_to_neighbor(head, next_pos)
        
        if direction is None:
            # This should never happen with a proper cycle, but fallback to A* just in case
            print(f"[Hamiltonian] Error: Cannot determine direction from {head} to {next_pos}")
            return self.get_next_move_astar()
        
        return direction
    
    def _build_hamiltonian_cycle(self):
        """
        Build a Hamiltonian cycle using AlphaPhoenix's zigzag pattern.
        
        For a 30x30 grid, this creates a pattern where:
        - Row 0 goes right-to-left (columns 29 down to 0)
        - Column 0 goes down (rows 1 to 29)
        - Odd columns go UP, even columns go DOWN
        - This forms a proper closed cycle
        
        The pattern ensures every cell is visited exactly once and the
        last cell connects back to the first cell.
        """
        self.hamiltonian_cycle = []
        self.cycle_index = {}
        
        # Create a 2D grid to hold the cycle indices
        # hamgrid[row][col] = index in the Hamiltonian cycle
        hamgrid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # First row: goes from right to left (nodenum down to nodenum-width+1)
        nodenum = GRID_HEIGHT * GRID_WIDTH  # Total cells = 900 for 30x30
        for col in range(GRID_WIDTH):
            hamgrid[0][col] = nodenum - col
        
        # For each column (except the first row)
        for col in range(GRID_WIDTH):
            if col % 2 == 0:
                # Even columns: go DOWN (rows 1 to GRID_HEIGHT-1)
                for row in range(1, GRID_HEIGHT):
                    hamgrid[row][col] = col * (GRID_HEIGHT - 1) - GRID_WIDTH + 2 + (row - 1)
            else:
                # Odd columns: go UP (rows GRID_HEIGHT-1 down to 1)
                for row in range(1, GRID_HEIGHT):
                    hamgrid[row][col] = col * (GRID_HEIGHT - 1) - GRID_WIDTH + 2 + (GRID_HEIGHT - 1 - row)
        
        # Convert hamgrid to a sorted list of (row, col) positions
        # Create list of (index, row, col) tuples
        index_to_pos = []
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                idx = hamgrid[row][col]
                index_to_pos.append((idx, row, col))
        
        # Sort by index to get the path order
        index_to_pos.sort()
        
        # Build the cycle list
        for idx, row, col in index_to_pos:
            pos = (row, col)
            self.cycle_index[pos] = len(self.hamiltonian_cycle)
            self.hamiltonian_cycle.append(pos)
        
        # Verify the cycle
        first_pos = self.hamiltonian_cycle[0]
        last_pos = self.hamiltonian_cycle[-1]
        
        print(f"[Hamiltonian] Built cycle: {len(self.hamiltonian_cycle)} cells")
        print(f"[Hamiltonian] Start: {first_pos}, End: {last_pos}")
        
        # Verify cycle closure (last position should be adjacent to first)
        row_diff = abs(first_pos[0] - last_pos[0])
        col_diff = abs(first_pos[1] - last_pos[1])
        is_closed = (row_diff + col_diff == 1)
        
        if is_closed:
            print(f"[Hamiltonian] Cycle is properly closed!")
        else:
            print(f"[Hamiltonian] WARNING: Cycle may not be closed properly (distance: {row_diff + col_diff})")
    
    def _get_direction_to_neighbor(self, from_pos, to_pos):
        """
        Get the direction needed to move from one position to an adjacent position.
        
        Args:
            from_pos: Starting position (row, col)
            to_pos: Target position (row, col)
        
        Returns:
            Direction tuple or None if positions are not adjacent
        """
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]
        
        # Check if positions are adjacent (Manhattan distance = 1)
        if abs(row_diff) + abs(col_diff) != 1:
            return None
        
        # Convert difference to direction
        if row_diff == -1 and col_diff == 0:
            return UP
        elif row_diff == 1 and col_diff == 0:
            return DOWN
        elif row_diff == 0 and col_diff == -1:
            return LEFT
        elif row_diff == 0 and col_diff == 1:
            return RIGHT
        
        return None
    
    def get_next_move_dhcr(self):
        """
        DHCR: Dynamic Hamiltonian Cycle with Shortcuts (AlphaPhoenix approach).
        
        This algorithm uses the Hamiltonian cycle as a safe fallback but takes
        A* shortcuts when it's safe to do so. Unlike the previous approach that
        re-evaluated every step, this implementation COMMITS to entire shortcut
        paths once validated, eliminating cumulative errors.
        
        Key differences from naive approach:
        1. Validates full A* path to food AND post-eating escape path
        2. Commits to entire shortcut atomically (no re-evaluation mid-path)
        3. Simulates virtual snake state after eating to verify safety
        4. Only uses Hamiltonian when no safe shortcut exists
        
        This combines the speed of A* with the guaranteed-win safety of Hamiltonian.
        Based on AlphaPhoenix's DHCR strategy with path commitment.
        
        Returns:
            tuple: Direction tuple (UP, DOWN, LEFT, or RIGHT)
        """
        # Build cycle if not already cached
        if self.hamiltonian_cycle is None:
            self._build_hamiltonian_cycle()
        
        head = self.game_engine.get_snake_head()
        food = self.game_engine.food
        
        # CRITICAL: If we have a committed path but food has changed, clear it!
        # This happens when we eat food - the path is no longer valid
        if self.committed_path and len(self.committed_path) > 0:
            path_target = self.committed_path[-1]  # Last position in path should be old food
            if path_target != food:
                # Food has changed (we ate it or it moved) - path is invalid
                self.committed_path = None
                self.committed_path_index = 0
        
        # Check if we're currently committed to a shortcut path
        if self.committed_path and self.committed_path_index < len(self.committed_path):
            # Verify we're still on the committed path (head should match expected position)
            expected_pos = self.committed_path[self.committed_path_index]
            
            if head == expected_pos:
                # We're on track - continue with committed path
                if self.committed_path_index + 1 < len(self.committed_path):
                    next_pos = self.committed_path[self.committed_path_index + 1]
                    direction = self._get_direction_from_positions(head, next_pos)
                    
                    if direction:
                        self.committed_path_index += 1
                        
                        # Check if we've completed the committed path (reached food)
                        if self.committed_path_index >= len(self.committed_path) - 1:
                            # Path complete - clear commitment
                            self.committed_path = None
                            self.committed_path_index = 0
                        
                        return direction
            
            # If we got here, something went wrong with the committed path
            # Clear it and revert to safe behavior
            self.committed_path = None
            self.committed_path_index = 0
        
        # No active commitment - evaluate if we should take a new shortcut
        shortcut_result = self._is_shortcut_safe(head, food)
        
        if shortcut_result:
            # Shortcut is safe - commit to the entire path
            safe_path = shortcut_result
            
            if len(safe_path) >= 2:
                # Commit to this path
                self.committed_path = safe_path
                self.committed_path_index = 0  # Start at index 0 (current head position)
                
                # Execute first move of the committed path
                next_pos = safe_path[1]
                direction = self._get_direction_from_positions(head, next_pos)
                
                if direction:
                    self.committed_path_index = 1  # Move to next position
                    return direction
        
        # No safe shortcut available - use Hamiltonian cycle
        # Clear any stale commitment
        self.committed_path = None
        self.committed_path_index = 0
        
        return self.get_next_move_hamiltonian()
    
    def _is_shortcut_safe(self, head, food):
        """
        Determine if taking an A* shortcut to food is safe (AlphaPhoenix approach).
        
        This uses a comprehensive validation strategy:
        1. Find A* path to food
        2. Verify path doesn't collide with snake body (accounting for tail movement)
        3. Simulate eating the food (virtual snake state)
        4. Verify we can escape from the post-eating position
        5. Ensure shortcut is actually shorter than Hamiltonian path
        
        The key innovation: we validate the ENTIRE journey (to food + escape)
        before committing, not just the path to food.
        
        Args:
            head: Current head position (row, col)
            food: Food position (row, col)
        
        Returns:
            list: The validated safe path if shortcut is safe, None otherwise
        """
        # 1. Find A* path to food
        path_to_food = self._find_path_astar(head, food)
        
        if not path_to_food or len(path_to_food) < 2:
            # No valid A* path to food
            return None
        
        # 2. Verify path doesn't collide with snake body
        if not self._validate_path_collision(path_to_food):
            return None
        
        # 3. Check if shortcut is actually beneficial
        if not self._is_shortcut_beneficial(head, food, path_to_food):
            return None
        
        # 4. CRITICAL: Simulate virtual state after eating food
        # This is the key to AlphaPhoenix's approach
        if not self._validate_post_eating_escape(path_to_food):
            return None
        
        # All checks passed - shortcut is safe!
        return path_to_food
    
    def _validate_path_collision(self, path):
        """
        Verify that following this path won't cause collisions with snake body.
        
        As the snake moves, the tail also moves, so we need to account for
        which body segments will still be present at each step.
        
        Args:
            path: List of positions from head to food
            
        Returns:
            bool: True if path is collision-free, False otherwise
        """
        snake_body = list(self.game_engine.snake)
        
        # Check each position in the path (skip position 0 which is current head)
        for i, pos in enumerate(path[1:], 1):
            # At step i, the tail will have moved i positions forward
            steps_taken = i
            
            # Body segments that will still be present
            # We haven't eaten yet, so tail keeps moving
            if steps_taken >= len(snake_body):
                # Entire snake has moved away
                continue
            
            # Segments still occupying space (head moves forward, tail moves too)
            remaining_body = set(snake_body[:-steps_taken]) if steps_taken > 0 else set(snake_body)
            
            # Check collision with remaining body
            if pos in remaining_body:
                return False
        
        return True
    
    def _is_shortcut_beneficial(self, head, food, path_to_food):
        """
        Check if the shortcut is actually shorter than the Hamiltonian path.
        
        Args:
            head: Current head position
            food: Food position  
            path_to_food: A* path from head to food
            
        Returns:
            bool: True if shortcut saves distance, False otherwise
        """
        # Verify positions are in cycle
        if head not in self.cycle_index or food not in self.cycle_index:
            return False
        
        head_idx = self.cycle_index[head]
        food_idx = self.cycle_index[food]
        cycle_len = len(self.hamiltonian_cycle)
        
        # Distance along Hamiltonian cycle
        cycle_distance = (food_idx - head_idx) % cycle_len
        
        # A* distance (path length - 1, since path includes start position)
        astar_distance = len(path_to_food) - 1
        
        # Dynamic threshold based on snake length
        snake_length = len(self.game_engine.snake)
        grid_size = GRID_WIDTH * GRID_HEIGHT
        snake_ratio = snake_length / grid_size
        
        # Conservative thresholds: longer snake = require more savings
        if snake_ratio > 0.7:
            # Snake occupies 70%+ of grid - very conservative
            threshold = 0.5  # Must save 50%+ distance
        elif snake_ratio > 0.5:
            # Snake occupies 50-70% of grid
            threshold = 0.65  # Must save 35%+ distance
        elif snake_ratio > 0.3:
            # Snake occupies 30-50% of grid
            threshold = 0.75  # Must save 25%+ distance
        else:
            # Snake is small (< 30% of grid)
            threshold = 0.85  # Must save 15%+ distance
        
        # Shortcut must be significantly shorter
        if astar_distance >= cycle_distance * threshold:
            return False
        
        return True
    
    def _validate_post_eating_escape(self, path_to_food):
        """
        CRITICAL VALIDATION: Simulate eating food and verify we can escape.
        
        This is AlphaPhoenix's key innovation: don't just check if we can reach
        the food - verify we can ESCAPE after eating it.
        
        Strategy:
        1. Simulate snake state after following path and eating food
        2. From that virtual state, verify we can reach a safe position
        3. Safe position = far enough along Hamiltonian cycle
        
        Args:
            path_to_food: A* path from current head to food
            
        Returns:
            bool: True if we can safely escape after eating, False otherwise
        """
        snake_body = list(self.game_engine.snake)
        astar_distance = len(path_to_food) - 1
        
        # Simulate snake state after following the path and eating food
        # New head will be at food position
        virtual_head = path_to_food[-1]
        
        # After eating, snake grows by 1
        # Old body moves along the path, but last segment doesn't disappear (growth)
        # Virtual body = path positions + tail segments that didn't move away
        
        # When we eat, we grow, so one less segment disappears from tail
        # Movement distance before eating: astar_distance - 1 (last step is eating)
        # After eating, tail has moved (astar_distance - 1) positions, then we grow
        virtual_body = list(path_to_food)  # Snake follows the path
        
        # Add tail segments that remain after movement
        # Tail moved (astar_distance - 1) positions, then we grew 1
        # Net tail movement: (astar_distance - 1) - 1 = astar_distance - 2
        tail_movement = astar_distance - 1  # Before eating
        segments_remaining = max(0, len(snake_body) - tail_movement)
        
        if segments_remaining > 0:
            # Add remaining tail segments
            virtual_body.extend(snake_body[-segments_remaining:])
        
        # Trim virtual body to correct length (old length + 1 from eating)
        expected_length = len(snake_body) + 1
        virtual_body = virtual_body[-expected_length:]
        
        # Now verify we can escape from virtual_head with virtual_body
        # Strategy: Try to reach a position well ahead on the Hamiltonian cycle
        
        if virtual_head not in self.cycle_index:
            return False
        
        virtual_head_idx = self.cycle_index[virtual_head]
        cycle_len = len(self.hamiltonian_cycle)
        
        # Look ahead on the cycle - we need to prove we can reach there
        # Use dynamic lookahead based on snake length
        lookahead = max(len(virtual_body) // 4, 20)
        target_idx = (virtual_head_idx + lookahead) % cycle_len
        target_pos = self.hamiltonian_cycle[target_idx]
        
        # Temporarily simulate the virtual state in game engine
        original_snake = list(self.game_engine.snake)
        
        try:
            # Set virtual state
            self.game_engine.snake.clear()
            self.game_engine.snake.extend(virtual_body)
            
            # Try to find escape path from virtual head to target
            escape_path = self._find_path_astar(virtual_head, target_pos)
            
            # Restore original state
            self.game_engine.snake.clear()
            self.game_engine.snake.extend(original_snake)
            
            if not escape_path:
                # Cannot escape after eating - shortcut is unsafe!
                return False
            
            # Additional check: ensure escape path is reasonable
            # If escape path is extremely long, we might be boxing ourselves in
            if len(escape_path) > lookahead * 2:
                # Escape path is too long - indicates potential trap
                return False
            
            # Verify the escape path itself doesn't have collisions
            # We need to account for the virtual body state
            self.game_engine.snake.clear()
            self.game_engine.snake.extend(virtual_body)
            
            escape_valid = self._validate_path_collision(escape_path)
            
            self.game_engine.snake.clear()
            self.game_engine.snake.extend(original_snake)
            
            if not escape_valid:
                return False
            
        except Exception as e:
            # If anything goes wrong, restore state and reject shortcut
            self.game_engine.snake.clear()
            self.game_engine.snake.extend(original_snake)
            return False
        
        # All escape checks passed!
        return True
