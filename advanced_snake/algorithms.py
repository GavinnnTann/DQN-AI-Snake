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
    
    def reset_dhcr_state(self):
        """
        Reset DHCR committed path state.
        Should be called when the game resets to clear stale path commitments.
        """
        self.committed_path = None
        self.committed_path_index = 0
        
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
        Find a path from start to end using A* algorithm with body-hugging strategy.
        
        Uses Manhattan distance as heuristic, with a body-hugging preference that:
        - Prefers paths that stay adjacent to the snake's body
        - Keeps the snake compact, preserving open space for maneuvering
        - Results in elegant coiling behavior around itself
        
        The hugging strategy helps DHCR by:
        - Making shortcuts that follow the body contour
        - Reducing fragmentation of free space
        - Creating more predictable, safer paths
        """
        # Priority queue for open set
        open_set = []
        # Using a counter to break ties consistently
        counter = 0
        
        # Get snake body for hugging calculations
        snake_body_set = set(self.game_engine.snake)
        snake_body_list = list(self.game_engine.snake)
        
        # Add start position to open set
        # Format: (f_score, hug_penalty, counter, position, path)
        # f_score = g_score (distance from start) + h_score (heuristic to end)
        # hug_penalty = negative means hugging (preferred), positive means not hugging
        heapq.heappush(open_set, (0, 0, counter, start, []))
        
        # Keep track of visited positions
        closed_set = set()
        
        while open_set:
            # Get position with lowest f_score (hug_penalty breaks ties)
            _, _, _, current, path = heapq.heappop(open_set)
            
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
                
                # BODY HUGGING: Calculate hugging bonus
                # Positions adjacent to body get a bonus (lower hug_penalty)
                hug_penalty = self._calculate_hug_penalty(neighbor, snake_body_set)
                
                # Add to open set with hug_penalty as secondary sort key
                counter += 1
                heapq.heappush(open_set, (f_score, hug_penalty, counter, neighbor, path + [current]))
        
        # No path found
        return None
    
    def _calculate_hug_penalty(self, position, snake_body_set):
        """
        Calculate a hugging penalty for a position.
        
        Lower penalty = more desirable (hugging the body)
        Higher penalty = less desirable (in open space)
        
        Strategy:
        - Positions adjacent to the snake body get a bonus (negative penalty)
        - Positions in open space get a penalty (positive)
        - More adjacent body segments = stronger hugging preference
        
        Args:
            position: The position to evaluate (row, col)
            snake_body_set: Set of all snake body positions
            
        Returns:
            float: Hugging penalty (lower = better for hugging)
        """
        row, col = position
        adjacent_body_count = 0
        
        # Check all 4 adjacent positions
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_row, adj_col = row + d_row, col + d_col
            adj_pos = (adj_row, adj_col)
            
            if adj_pos in snake_body_set:
                adjacent_body_count += 1
        
        # Also check diagonal positions for "corner hugging"
        for d_row, d_col in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            adj_row, adj_col = row + d_row, col + d_col
            adj_pos = (adj_row, adj_col)
            
            if adj_pos in snake_body_set:
                adjacent_body_count += 0.5  # Diagonals count less
        
        # Convert to penalty: more adjacent body = lower (better) penalty
        # Range: -3 (hugging tightly) to 0 (no body nearby)
        hug_penalty = -adjacent_body_count * 0.5
        
        # Also consider wall hugging as a secondary preference
        # Walls provide structure similar to body
        wall_adjacent = 0
        if row == 0 or row == GRID_HEIGHT - 1:
            wall_adjacent += 1
        if col == 0 or col == GRID_WIDTH - 1:
            wall_adjacent += 1
        
        hug_penalty -= wall_adjacent * 0.25
        
        return hug_penalty
    
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
        """
        Get a safe move when normal pathfinding fails.
        
        Enhanced to consider:
        1. Escape routes (avoid dead ends)
        2. Hamiltonian cycle ordering when available
        3. Movement towards food as tie-breaker
        """
        head = self.game_engine.get_snake_head()
        snake = list(self.game_engine.snake)
        snake_body_set = set(snake[:-1])  # Exclude tail
        
        valid_moves = self.game_engine.get_valid_moves()
        
        if not valid_moves:
            return self.game_engine.direction
        
        # Score each move
        scored_moves = []
        
        for move in valid_moves:
            new_row = head[0] + move[0]
            new_col = head[1] + move[1]
            
            if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                continue
                
            new_pos = (new_row, new_col)
            
            # Count escape routes
            escape_count = self._count_escape_routes(new_pos, snake_body_set)
            
            # Check Hamiltonian cycle ordering if available
            cycle_score = 0
            if self.hamiltonian_cycle and new_pos in self.cycle_index:
                tail = snake[-1]
                if tail in self.cycle_index:
                    new_idx = self.cycle_index[new_pos]
                    tail_idx = self.cycle_index[tail]
                    cycle_len = len(self.hamiltonian_cycle)
                    gap = (tail_idx - new_idx - 1) % cycle_len
                    # Prefer moves that maintain good gap
                    cycle_score = min(gap, 100)  # Cap at 100
            
            # Food proximity bonus
            food = self.game_engine.food
            dist_to_food = abs(new_pos[0] - food[0]) + abs(new_pos[1] - food[1])
            food_score = 100 - min(dist_to_food, 100)
            
            # Combined score
            score = escape_count * 1000 + cycle_score * 10 + food_score
            
            scored_moves.append((score, move, escape_count))
        
        if not scored_moves:
            return valid_moves[0] if valid_moves else self.game_engine.direction
        
        # Sort by score (higher is better)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        return scored_moves[0][1]
    
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
        DYNAMIC Hamiltonian cycle algorithm.
        
        Instead of blindly following a fixed path, this uses the Hamiltonian cycle
        as an ORDERING SYSTEM. The key insight (from AlphaPhoenix):
        
        - Every cell has an index in the cycle (0 to N-1)
        - The snake's head must ALWAYS stay "ahead" of its tail in cycle order
        - As long as: head_index > tail_index (mod cycle_length), we're safe
        
        This allows us to:
        1. Take any move that keeps head ahead of tail
        2. Skip forward in the cycle (shortcuts) when safe
        3. Never get trapped because we always have a path to our tail
        
        Returns:
            tuple: Direction tuple (UP, DOWN, LEFT, or RIGHT)
        """
        # Build cycle if not already cached
        if self.hamiltonian_cycle is None:
            self._build_hamiltonian_cycle()
        
        head = self.game_engine.get_snake_head()
        snake = list(self.game_engine.snake)
        tail = snake[-1]
        
        # Find current positions in cycle
        if head not in self.cycle_index:
            print(f"[Hamiltonian] Warning: Head at {head} not in cycle")
            return self._get_safe_move()
        
        if tail not in self.cycle_index:
            print(f"[Hamiltonian] Warning: Tail at {tail} not in cycle")
            return self._get_safe_move()
        
        head_idx = self.cycle_index[head]
        tail_idx = self.cycle_index[tail]
        cycle_len = len(self.hamiltonian_cycle)
        
        # Calculate the "gap" - free space between head and tail in cycle order
        # This is how many cells we can potentially move through
        gap = (tail_idx - head_idx - 1) % cycle_len
        
        # Get all valid moves (not blocked by body except tail)
        snake_body_set = set(snake[:-1])  # Exclude tail, it will move
        
        # Evaluate each possible move
        candidates = []
        emergency_candidates = []  # Moves that are safe from collision but might not satisfy gap
        
        for direction in [UP, DOWN, LEFT, RIGHT]:
            new_row = head[0] + direction[0]
            new_col = head[1] + direction[1]
            
            # Check bounds
            if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                continue
            
            new_pos = (new_row, new_col)
            
            # Check if blocked by body (tail is OK, it moves)
            if new_pos in snake_body_set:
                continue
            
            # Get cycle index of new position
            if new_pos not in self.cycle_index:
                continue
            
            new_idx = self.cycle_index[new_pos]
            
            # Calculate how far ahead of tail we'd be after this move
            new_gap = (tail_idx - new_idx - 1) % cycle_len
            
            # We need to leave room for the entire snake body
            min_safe_gap = len(snake) - 1
            
            # Count escape routes for scoring
            escape_count = self._count_escape_routes(new_pos, snake_body_set)
            
            # Check if this move follows the natural cycle order
            is_natural_next = (new_idx == (head_idx + 1) % cycle_len)
            
            # Score: prefer natural cycle order, but allow skips if safe
            score = (
                escape_count * 100 +
                (10 if is_natural_next else 0) +
                new_gap
            )
            
            if new_gap >= min_safe_gap:
                # Ideal case: satisfies gap requirement
                candidates.append((score, direction, new_pos, new_gap, is_natural_next))
            else:
                # Emergency: doesn't satisfy gap but isn't blocked
                # This can happen in tight situations - still better than dying
                emergency_candidates.append((score, direction, new_pos, new_gap, escape_count))
        
        if candidates:
            # We have ideal candidates - use them
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        if emergency_candidates:
            # No ideal candidates, but we have emergency moves
            # Pick the one with the most escape routes (best survival chance)
            emergency_candidates.sort(key=lambda x: (x[4], x[3]), reverse=True)
            best_emergency = emergency_candidates[0]
            # Only use emergency if it has escape routes
            if best_emergency[4] > 0:  # escape_count > 0
                return best_emergency[1]
        
        # Truly no safe moves - fall back to any valid move
        print(f"[Hamiltonian] No safe moves! Head at {head}, gap={gap}")
        return self._get_safe_move()
    
    def _get_dynamic_hamiltonian_move(self, target_pos=None):
        """
        Get a move that follows dynamic Hamiltonian ordering while optionally
        moving towards a target position.
        
        This is used by DHCR to find safe paths that respect cycle ordering.
        
        Args:
            target_pos: Optional target to move towards (e.g., food)
            
        Returns:
            Direction tuple or None
        """
        if self.hamiltonian_cycle is None:
            self._build_hamiltonian_cycle()
        
        head = self.game_engine.get_snake_head()
        snake = list(self.game_engine.snake)
        tail = snake[-1]
        
        head_idx = self.cycle_index.get(head)
        tail_idx = self.cycle_index.get(tail)
        
        if head_idx is None or tail_idx is None:
            return None
        
        cycle_len = len(self.hamiltonian_cycle)
        snake_body_set = set(snake[:-1])
        min_safe_gap = len(snake) - 1
        
        candidates = []
        
        for direction in [UP, DOWN, LEFT, RIGHT]:
            new_row = head[0] + direction[0]
            new_col = head[1] + direction[1]
            
            if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                continue
            
            new_pos = (new_row, new_col)
            
            if new_pos in snake_body_set:
                continue
            
            new_idx = self.cycle_index.get(new_pos)
            if new_idx is None:
                continue
            
            new_gap = (tail_idx - new_idx - 1) % cycle_len
            
            if new_gap < min_safe_gap:
                continue
            
            # Score based on target proximity if provided
            if target_pos:
                dist_to_target = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])
                score = -dist_to_target  # Negative because lower distance is better
            else:
                score = new_gap  # Prefer maintaining large gap
            
            candidates.append((score, direction, new_pos, new_gap))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _find_safe_cycle_adjacent_move(self, head, current_cycle_idx, blocked_positions):
        """
        Find a safe move when the normal Hamiltonian cycle path is blocked.
        
        Strategy: Look for adjacent positions that are:
        1. Not blocked by the snake body
        2. Can reach a position further ahead in the Hamiltonian cycle
        3. Won't trap us in a dead end
        
        Args:
            head: Current head position
            current_cycle_idx: Current index in the Hamiltonian cycle
            blocked_positions: Set/list of positions occupied by snake body
            
        Returns:
            Direction tuple or None if no safe move found
        """
        head_row, head_col = head
        cycle_len = len(self.hamiltonian_cycle)
        snake_length = len(self.game_engine.snake)
        
        # Get all adjacent positions with their cycle indices
        candidates = []
        for direction in [UP, DOWN, LEFT, RIGHT]:
            new_row = head_row + direction[0]
            new_col = head_col + direction[1]
            
            # Check bounds
            if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                continue
            
            new_pos = (new_row, new_col)
            
            # Check if blocked
            if new_pos in blocked_positions:
                continue
            
            # Get cycle index of this position
            if new_pos in self.cycle_index:
                pos_cycle_idx = self.cycle_index[new_pos]
                
                # Calculate how far ahead this position is in the cycle
                cycle_distance = (pos_cycle_idx - current_cycle_idx) % cycle_len
                
                # CRITICAL: Check if this move has an escape route
                # Count how many further moves are available from this position
                escape_count = self._count_escape_routes(new_pos, blocked_positions)
                
                # Score: prioritize moves with more escape routes, then cycle distance
                # Escape routes are weighted heavily to avoid traps
                score = (escape_count * 1000) - cycle_distance
                
                candidates.append((score, direction, new_pos, escape_count))
        
        if not candidates:
            return None
        
        # Sort by score (higher is better - more escape routes)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Verify the best candidate can actually reach a safe position on the cycle
        # Try to find a path to a position ahead in the cycle
        for score, direction, new_pos, escape_count in candidates:
            if escape_count > 0:  # At least one escape route
                # Check if we can reach a position ahead in the cycle from here
                if self._can_reach_cycle_ahead(new_pos, current_cycle_idx, blocked_positions):
                    return direction
        
        # If no candidate can definitively reach ahead, pick the one with most escapes
        if candidates and candidates[0][3] > 0:
            return candidates[0][1]
        
        return None
    
    def _count_escape_routes(self, position, blocked_positions):
        """
        Count how many valid moves are available from a position.
        This helps identify dead ends.
        """
        row, col = position
        count = 0
        
        for direction in [UP, DOWN, LEFT, RIGHT]:
            new_row = row + direction[0]
            new_col = col + direction[1]
            
            # Check bounds
            if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                continue
            
            new_pos = (new_row, new_col)
            
            # The tail will move when we move, so we need to consider that
            # For simplicity, just check if position is currently blocked
            # (excluding tail which will move)
            snake_body = list(self.game_engine.snake)
            body_without_tail = snake_body[:-1] if len(snake_body) > 1 else snake_body
            
            if new_pos not in body_without_tail:
                count += 1
        
        return count
    
    def _can_reach_cycle_ahead(self, start_pos, current_cycle_idx, blocked_positions):
        """
        Check if we can reach a position that's ahead in the Hamiltonian cycle.
        Uses a limited BFS to find a path.
        """
        cycle_len = len(self.hamiltonian_cycle)
        snake_length = len(self.game_engine.snake)
        
        # Target: any position that's ahead in the cycle by at least a few steps
        # We want to skip past where the snake body is blocking
        min_ahead = min(snake_length + 5, cycle_len // 4)
        
        # BFS with limited depth to avoid expensive searches
        max_depth = min(50, cycle_len // 4)
        
        visited = {start_pos}
        queue = [(start_pos, 0)]  # (position, depth)
        
        # Simulate snake body moving as we explore
        snake_body = list(self.game_engine.snake)
        
        while queue:
            pos, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            row, col = pos
            
            for direction in [UP, DOWN, LEFT, RIGHT]:
                new_row = row + direction[0]
                new_col = col + direction[1]
                
                # Check bounds
                if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                    continue
                
                new_pos = (new_row, new_col)
                
                if new_pos in visited:
                    continue
                
                # Check if blocked (accounting for tail movement)
                steps = depth + 1
                if steps < len(snake_body):
                    body_at_step = snake_body[:-steps]
                else:
                    body_at_step = []
                
                if new_pos in body_at_step:
                    continue
                
                visited.add(new_pos)
                
                # Check if this position is sufficiently ahead in cycle
                if new_pos in self.cycle_index:
                    pos_cycle_idx = self.cycle_index[new_pos]
                    distance_ahead = (pos_cycle_idx - current_cycle_idx) % cycle_len
                    
                    if distance_ahead >= min_ahead and distance_ahead < cycle_len - snake_length:
                        # Found a reachable position that's ahead in the cycle
                        return True
                
                queue.append((new_pos, depth + 1))
        
        return False
    
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
                    
                    # CRITICAL SAFETY CHECK: Verify next position is still safe
                    # (not colliding with current snake body)
                    snake_body = list(self.game_engine.snake)
                    # Check against body excluding tail (tail will move)
                    body_to_check = snake_body[:-1] if len(snake_body) > 1 else snake_body
                    if next_pos in body_to_check:
                        # Next position would cause collision! Abort shortcut.
                        self.committed_path = None
                        self.committed_path_index = 0
                        # Fall through to Hamiltonian cycle
                    else:
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
        Determine if taking an A* shortcut to food is safe using DYNAMIC cycle ordering.
        
        The key insight (AlphaPhoenix): instead of checking if we can "return to 
        the fixed cycle path", we verify that taking this shortcut maintains
        proper HEAD-TAIL ORDERING in the cycle.
        
        Safety rule: head_cycle_index must always stay "ahead" of tail_cycle_index
        (with enough gap for the snake body).
        
        Strategy:
        1. Find A* path to food
        2. Verify path doesn't collide with snake body
        3. Check if shortcut maintains safe cycle ordering
        4. Simulate post-eating state and verify continued safety
        
        Args:
            head: Current head position (row, col)
            food: Food position (row, col)
        
        Returns:
            list: The validated safe path if shortcut is safe, None otherwise
        """
        # 1. Find A* path to food (with hugging)
        path_to_food = self._find_path_astar(head, food)
        
        if not path_to_food or len(path_to_food) < 2:
            return None
        
        # 2. Verify path doesn't collide with snake body
        if not self._validate_path_collision(path_to_food):
            return None
        
        # 3. Check if shortcut maintains safe cycle ordering
        if not self._validate_shortcut_cycle_ordering(path_to_food):
            return None
        
        # 4. Check if shortcut is beneficial (not too risky)
        if not self._is_shortcut_beneficial(head, food, path_to_food):
            return None
        
        # 5. Validate post-eating escape using dynamic ordering
        if not self._validate_post_eating_dynamic(path_to_food):
            return None
        
        return path_to_food
    
    def _validate_shortcut_cycle_ordering(self, path):
        """
        Validate that taking this shortcut path maintains safe cycle ordering.
        
        The key invariant: at every step of the path, the head must maintain
        a safe "gap" ahead of the tail in cycle order.
        
        Args:
            path: List of positions from head to food
            
        Returns:
            bool: True if cycle ordering is maintained throughout, False otherwise
        """
        if self.hamiltonian_cycle is None:
            self._build_hamiltonian_cycle()
        
        snake = list(self.game_engine.snake)
        cycle_len = len(self.hamiltonian_cycle)
        
        # Simulate snake movement along the path
        simulated_snake = list(snake)
        
        for i in range(1, len(path)):
            next_pos = path[i]
            is_eating = (i == len(path) - 1)
            
            # Move head to next position
            simulated_snake.insert(0, next_pos)
            
            if not is_eating:
                simulated_snake.pop()  # Tail moves
            # else: eating, snake grows
            
            # Check cycle ordering
            new_head = simulated_snake[0]
            new_tail = simulated_snake[-1]
            
            head_idx = self.cycle_index.get(new_head)
            tail_idx = self.cycle_index.get(new_tail)
            
            if head_idx is None or tail_idx is None:
                return False
            
            # Calculate gap between tail and head
            gap = (tail_idx - head_idx - 1) % cycle_len
            min_safe_gap = len(simulated_snake) - 1
            
            if gap < min_safe_gap:
                # This path would violate cycle ordering - head catching up to tail!
                return False
        
        return True
    
    def _validate_post_eating_dynamic(self, path_to_food):
        """
        Validate post-eating safety using dynamic cycle ordering.
        
        After eating, verify that the snake can continue to make moves that
        maintain proper cycle ordering (head ahead of tail).
        
        Args:
            path_to_food: Path from current head to food
            
        Returns:
            bool: True if post-eating state allows safe continuation
        """
        if self.hamiltonian_cycle is None:
            self._build_hamiltonian_cycle()
        
        # Simulate snake state after following path and eating
        snake = list(self.game_engine.snake)
        simulated_snake = list(snake)
        
        for i in range(1, len(path_to_food)):
            next_pos = path_to_food[i]
            is_eating = (i == len(path_to_food) - 1)
            
            simulated_snake.insert(0, next_pos)
            if not is_eating:
                simulated_snake.pop()
        
        # Now check: from this post-eating state, do we have valid moves?
        virtual_head = simulated_snake[0]
        virtual_tail = simulated_snake[-1]
        virtual_body_set = set(simulated_snake[:-1])  # Exclude tail
        
        head_idx = self.cycle_index.get(virtual_head)
        tail_idx = self.cycle_index.get(virtual_tail)
        
        if head_idx is None or tail_idx is None:
            return False
        
        cycle_len = len(self.hamiltonian_cycle)
        min_safe_gap = len(simulated_snake) - 1
        
        # Check if at least one valid move exists
        valid_moves_exist = False
        
        for direction in [UP, DOWN, LEFT, RIGHT]:
            new_row = virtual_head[0] + direction[0]
            new_col = virtual_head[1] + direction[1]
            
            if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                continue
            
            new_pos = (new_row, new_col)
            
            if new_pos in virtual_body_set:
                continue
            
            new_idx = self.cycle_index.get(new_pos)
            if new_idx is None:
                continue
            
            # Check if this move maintains safe gap
            new_gap = (tail_idx - new_idx - 1) % cycle_len
            
            if new_gap >= min_safe_gap:
                valid_moves_exist = True
                break
        
        if not valid_moves_exist:
            return False
        
        # Additional check: simulate several more moves to ensure we're not trapped
        # This catches cases where we have one move but then get stuck
        test_snake = list(simulated_snake)
        
        for _ in range(min(20, len(simulated_snake))):
            test_head = test_snake[0]
            test_tail = test_snake[-1]
            test_body_set = set(test_snake[:-1])
            
            test_head_idx = self.cycle_index.get(test_head)
            test_tail_idx = self.cycle_index.get(test_tail)
            
            if test_head_idx is None or test_tail_idx is None:
                return False
            
            # Find best valid move (prefer cycle order)
            best_move = None
            best_gap = -1
            
            for direction in [UP, DOWN, LEFT, RIGHT]:
                new_row = test_head[0] + direction[0]
                new_col = test_head[1] + direction[1]
                
                if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                    continue
                
                new_pos = (new_row, new_col)
                
                if new_pos in test_body_set:
                    continue
                
                new_idx = self.cycle_index.get(new_pos)
                if new_idx is None:
                    continue
                
                new_gap = (test_tail_idx - new_idx - 1) % cycle_len
                min_gap = len(test_snake) - 1
                
                if new_gap >= min_gap and new_gap > best_gap:
                    best_gap = new_gap
                    best_move = new_pos
            
            if best_move is None:
                # Got stuck during simulation - shortcut is unsafe
                return False
            
            # Simulate move (no eating, just moving)
            test_snake.insert(0, best_move)
            test_snake.pop()
        
        return True
    
    def _validate_path_collision(self, path):
        """
        Verify that following this path won't cause collisions with snake body.
        
        CRITICAL: Properly simulate the snake's body as it follows the path.
        The snake's body trails behind the head - as the head moves forward,
        the body occupies the head's previous positions, and the tail advances.
        
        Args:
            path: List of positions from head to food
            
        Returns:
            bool: True if path is collision-free, False otherwise
        """
        # Simulate the snake body step by step as it follows the path
        # Start with current snake state
        simulated_body = list(self.game_engine.snake)
        
        # Check each step of the path (skip position 0 which is current head)
        for i in range(1, len(path)):
            next_pos = path[i]
            
            # Check if next position collides with the current simulated body
            # The tail will move away on this step (we haven't eaten yet)
            # So we check collision with body excluding tail
            if len(simulated_body) > 1:
                body_without_tail = simulated_body[:-1]
            else:
                body_without_tail = simulated_body
            
            if next_pos in body_without_tail:
                return False
            
            # Simulate the snake moving: add new head, remove tail
            simulated_body.insert(0, next_pos)
            simulated_body.pop()  # Tail moves away (not eating yet)
        
        return True
    
    def _is_shortcut_beneficial(self, head, food, path_to_food):
        """
        Check if the shortcut is actually shorter than the Hamiltonian path.
        Also performs critical safety checks to ensure shortcut doesn't trap the snake.
        
        Args:
            head: Current head position
            food: Food position  
            path_to_food: A* path from head to food
            
        Returns:
            bool: True if shortcut saves distance AND is safe, False otherwise
        """
        # Verify positions are in cycle
        if head not in self.cycle_index or food not in self.cycle_index:
            return False
        
        head_idx = self.cycle_index[head]
        food_idx = self.cycle_index[food]
        cycle_len = len(self.hamiltonian_cycle)
        
        # Distance along Hamiltonian cycle (forward direction only)
        cycle_distance = (food_idx - head_idx) % cycle_len
        
        # A* distance (path length - 1, since path includes start position)
        astar_distance = len(path_to_food) - 1
        
        # Dynamic threshold based on snake length
        snake_length = len(self.game_engine.snake)
        grid_size = GRID_WIDTH * GRID_HEIGHT
        snake_ratio = snake_length / grid_size
        
        # CRITICAL SAFETY CHECK: Don't take shortcuts when snake is very long
        # At this point, Hamiltonian cycle is the safest path
        if snake_ratio > 0.75:
            # Snake occupies 75%+ of grid - NO shortcuts, too risky
            return False
        
        # Conservative thresholds: longer snake = require more savings
        if snake_ratio > 0.6:
            # Snake occupies 60-75% of grid - very conservative
            threshold = 0.4  # Must save 60%+ distance
        elif snake_ratio > 0.45:
            # Snake occupies 45-60% of grid
            threshold = 0.55  # Must save 45%+ distance
        elif snake_ratio > 0.3:
            # Snake occupies 30-45% of grid
            threshold = 0.7  # Must save 30%+ distance
        else:
            # Snake is small (< 30% of grid)
            threshold = 0.85  # Must save 15%+ distance
        
        # Shortcut must be significantly shorter
        if astar_distance >= cycle_distance * threshold:
            return False
        
        # CRITICAL: Check that the shortcut doesn't "cut across" the snake's tail
        # in the Hamiltonian cycle ordering. This prevents trapping ourselves.
        tail_idx = self.cycle_index.get(self.game_engine.snake[-1])
        if tail_idx is not None:
            # Check if food is between head and tail in cycle order
            # If head -> food -> tail in cycle order, shortcut might cut off tail
            if head_idx < tail_idx:
                # Normal case: head is before tail in cycle
                if head_idx < food_idx < tail_idx:
                    # Food is between head and tail - could be dangerous
                    # Only allow if shortcut is VERY significant
                    if astar_distance > cycle_distance * 0.3:
                        return False
            else:
                # Wrapped case: head is after tail (cycle wraps around)
                if food_idx > head_idx or food_idx < tail_idx:
                    # Food is in the "forward" region - could cut off tail
                    if astar_distance > cycle_distance * 0.3:
                        return False
        
        return True
    
    def _validate_post_eating_escape(self, path_to_food):
        """
        CRITICAL VALIDATION: Simulate eating food and verify we can safely
        rejoin the Hamiltonian cycle without getting trapped.
        
        This is the key to preventing deadlocks: we don't just check if we can
        "escape" - we verify we can cleanly rejoin the Hamiltonian cycle and
        continue following it without getting blocked.
        
        Strategy:
        1. Simulate snake state after following path and eating food
        2. Check if the immediate Hamiltonian cycle path is clear
        3. If not, verify we can navigate back to a safe cycle position
        4. Simulate several steps of Hamiltonian cycle following to ensure no blocks
        
        Args:
            path_to_food: A* path from current head to food
            
        Returns:
            bool: True if we can safely rejoin cycle after eating, False otherwise
        """
        original_snake = list(self.game_engine.snake)
        
        # Simulate snake movement step by step along the path
        simulated_body = list(original_snake)
        
        for i in range(1, len(path_to_food)):
            next_pos = path_to_food[i]
            is_eating_food = (i == len(path_to_food) - 1)  # Last step = eating
            
            # Move head to next position
            simulated_body.insert(0, next_pos)
            
            if not is_eating_food:
                # Not eating yet, tail moves
                simulated_body.pop()
            # else: eating food, snake grows, tail stays
        
        # After simulation, virtual_head is at food position
        virtual_head = path_to_food[-1]
        virtual_body = simulated_body
        
        if virtual_head not in self.cycle_index:
            return False
        
        virtual_head_idx = self.cycle_index[virtual_head]
        cycle_len = len(self.hamiltonian_cycle)
        
        # CRITICAL: Simulate following the Hamiltonian cycle for several steps
        # and verify we don't get blocked at any point
        # We need to simulate long enough for the tail to clear any blocking positions
        
        steps_to_simulate = min(len(virtual_body) + 10, cycle_len // 2)
        test_body = list(virtual_body)
        current_idx = virtual_head_idx
        
        for step in range(steps_to_simulate):
            # Get next position in cycle
            next_idx = (current_idx + 1) % cycle_len
            next_pos = self.hamiltonian_cycle[next_idx]
            
            # Check if next position is blocked by body (excluding tail)
            body_without_tail = test_body[:-1] if len(test_body) > 1 else test_body
            
            if next_pos in body_without_tail:
                # Cycle is blocked! Check if we can find an alternative path
                # that eventually rejoins the cycle
                
                # Try to find ANY path to a position ahead in the cycle
                # that avoids the current body
                escape_found = False
                
                # Look for positions ahead in cycle that we could reach
                for lookahead in range(10, min(100, cycle_len // 4), 10):
                    target_idx = (current_idx + lookahead) % cycle_len
                    target_pos = self.hamiltonian_cycle[target_idx]
                    
                    # Check if target is reachable with current body state
                    # Use a simplified reachability check
                    if self._is_position_reachable_with_body(
                        test_body[0], target_pos, set(body_without_tail)
                    ):
                        # Can reach this position - but verify the path there
                        # won't trap us further
                        
                        # Simulate moving to target and continuing cycle
                        steps_to_target = lookahead
                        future_body = list(test_body)
                        
                        # Rough simulation: move tail for each step
                        for _ in range(min(steps_to_target, len(future_body) - 1)):
                            future_body.pop()
                        future_body.insert(0, target_pos)
                        
                        # Check if cycle continues from there
                        future_next_idx = (target_idx + 1) % cycle_len
                        future_next_pos = self.hamiltonian_cycle[future_next_idx]
                        
                        if future_next_pos not in future_body[:-1]:
                            escape_found = True
                            break
                
                if not escape_found:
                    # Cannot escape this blocked position - shortcut is unsafe
                    return False
                
                # Found an escape, but this shortcut is risky
                # Be extra conservative - reject if snake is long
                snake_ratio = len(virtual_body) / (GRID_WIDTH * GRID_HEIGHT)
                if snake_ratio > 0.3:
                    return False
                
                # Accept the shortcut but note it required escape planning
                break
            
            # Simulate move: add new head, remove tail
            test_body.insert(0, next_pos)
            test_body.pop()
            current_idx = next_idx
        
        # Additional safety: verify we haven't created a situation where
        # the snake body is spread across the cycle in a way that fragments
        # the available path
        if not self._verify_cycle_connectivity(virtual_body, virtual_head_idx):
            return False
        
        return True
    
    def _is_position_reachable_with_body(self, start, target, blocked_set):
        """
        Quick BFS check if target is reachable from start avoiding blocked positions.
        Uses limited depth for performance.
        """
        if start == target:
            return True
        
        max_depth = 50
        visited = {start}
        queue = [(start, 0)]
        
        while queue:
            pos, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            row, col = pos
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                
                if new_row < 0 or new_row >= GRID_HEIGHT or new_col < 0 or new_col >= GRID_WIDTH:
                    continue
                
                new_pos = (new_row, new_col)
                
                if new_pos in visited or new_pos in blocked_set:
                    continue
                
                if new_pos == target:
                    return True
                
                visited.add(new_pos)
                queue.append((new_pos, depth + 1))
        
        return False
    
    def _verify_cycle_connectivity(self, snake_body, head_cycle_idx):
        """
        Verify that the snake body doesn't fragment the Hamiltonian cycle
        in a way that would prevent smooth traversal.
        
        Key insight: If the snake body spans too much of the cycle continuously,
        we might not have room to maneuver if the cycle gets blocked.
        
        Returns:
            bool: True if cycle connectivity is acceptable, False if fragmented badly
        """
        cycle_len = len(self.hamiltonian_cycle)
        body_set = set(snake_body)
        
        # Find the span of cycle indices occupied by the snake body
        body_indices = []
        for pos in snake_body:
            if pos in self.cycle_index:
                body_indices.append(self.cycle_index[pos])
        
        if not body_indices:
            return True
        
        # Check what fraction of the "forward" cycle is blocked
        # Forward = from head to tail in cycle order
        head_idx = head_cycle_idx
        
        # Count consecutive blocked positions ahead
        consecutive_blocked = 0
        max_consecutive = 0
        
        for i in range(1, cycle_len):
            check_idx = (head_idx + i) % cycle_len
            check_pos = self.hamiltonian_cycle[check_idx]
            
            if check_pos in body_set:
                consecutive_blocked += 1
                max_consecutive = max(max_consecutive, consecutive_blocked)
            else:
                consecutive_blocked = 0
        
        # If there's a very long consecutive blocked section, it's dangerous
        # because we'd have to navigate around it
        snake_ratio = len(snake_body) / cycle_len
        
        if snake_ratio > 0.4:
            # Long snake - be very careful about fragmentation
            # Allow at most snake_length consecutive blocks
            if max_consecutive > len(snake_body) * 0.8:
                return False
        
        return True
