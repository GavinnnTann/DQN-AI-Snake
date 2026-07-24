"""
Safety shield for the DQN snake agent (Path A: "trained snake + survival guarantee").

The DQN proposes a preference over the 3 relative moves (turn-right / straight /
turn-left).  This module vetoes moves that would kill or trap the snake and picks
the best *survivable* move according to the DQN's preference.

Core ideas
----------
1. Simulate each candidate move on a copy of the board (never mutate game state).
2. Reject moves that hit a wall or the body.
3. Among the survivors, strongly prefer moves after which the head can still
   reach the tail ("tail-reachability").  Keeping a head->tail path is the classic
   invariant that lets a snake survive indefinitely - if you can always reach your
   tail, the tail keeps vacating cells and you never get sealed in.
4. In the endgame (board mostly full) bias toward a fixed Hamiltonian cycle so the
   snake fills the remaining cells in an order that is guaranteed collision-free,
   which is how the game actually gets *won*.
5. If no move keeps a tail path (forced), take the move that leaves the most open
   space so we survive as long as possible.

The shield is deliberately independent of the network's internal state
representation, so it works with any existing model without retraining.
"""

from collections import deque
from constants import (GRID_HEIGHT, GRID_WIDTH, UP, DOWN, LEFT, RIGHT,
                      INITIAL_SNAKE_LENGTH)

_ORTHO = (UP, DOWN, LEFT, RIGHT)


def relative_to_absolute(current_dir, action):
    """Map a relative action to an absolute direction.

    Convention (MUST match EnhancedDQNAgent.perform_action):
        0 = turn right, 1 = straight, 2 = turn left
    """
    if current_dir == UP:
        return (RIGHT, UP, LEFT)[action]
    elif current_dir == RIGHT:
        return (DOWN, RIGHT, UP)[action]
    elif current_dir == DOWN:
        return (LEFT, DOWN, RIGHT)[action]
    else:  # LEFT
        return (UP, LEFT, DOWN)[action]


def simulate_move(snake_list, food, direction, height, width):
    """Apply a move to a copy of the snake.

    Args:
        snake_list: list of (row, col), head first.
        food: (row, col) of the food.
        direction: absolute direction tuple.
    Returns:
        (new_body, ate, alive)
        new_body is None when the move is fatal.
    """
    head = snake_list[0]
    nr, nc = head[0] + direction[0], head[1] + direction[1]

    # Wall collision.
    if nr < 0 or nr >= height or nc < 0 or nc >= width:
        return None, False, False

    new_head = (nr, nc)
    ate = (new_head == food)

    if ate:
        # Tail does not move this step, so the whole body is solid.
        if new_head in snake_list:
            return None, True, False
        new_body = [new_head] + snake_list
    else:
        # Tail moves away, so stepping onto the current tail cell is legal.
        if new_head in snake_list[:-1]:
            return None, False, False
        new_body = [new_head] + snake_list[:-1]

    return new_body, ate, True


def analyze_body(new_body, height, width):
    """Flood-fill from the head of ``new_body``.

    Returns:
        (reaches_tail, free_space)
        reaches_tail: can the head still reach the tail cell through open space?
        free_space:   number of empty cells reachable from the head.
    """
    body_set = set(new_body)
    head = new_body[0]
    tail = new_body[-1]

    # The tail cell will vacate next step, so treat it as walkable / a goal.
    blocked = body_set
    reaches_tail = False

    seen = {head}
    frontier = deque((head,))
    free_space = 0

    while frontier:
        r, c = frontier.popleft()
        free_space += 1
        for dr, dc in _ORTHO:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                p = (nr, nc)
                if p in seen:
                    continue
                if p == tail:
                    reaches_tail = True
                    # Do not expand through the tail; it is the goal, not open space.
                    seen.add(p)
                    continue
                if p not in blocked:
                    seen.add(p)
                    frontier.append(p)

    return reaches_tail, free_space


class HamiltonianCycle:
    """A fixed Hamiltonian cycle over the grid (requires an even number of rows).

    Used only as an endgame / last-resort guide: following the cycle visits every
    cell exactly once and returns to start, so it fills the board without self-
    collision.  We still validate the suggested move through the shield, so a
    mismatched body never causes a crash.
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.valid = (height % 2 == 0)
        self._next = {}
        if self.valid:
            self._build()

    def _successor(self, r, c):
        H, W = self.height, self.width
        if c == 0:
            # Left spine: travel up; leave at the top-left corner.
            if r == 0:
                return (0, 1)
            return (r - 1, 0)
        # Field columns 1..W-1: horizontal serpentine.
        if r % 2 == 0:
            # Even rows move right; drop down at the right wall.
            if c == W - 1:
                return (r + 1, c)
            return (r, c + 1)
        else:
            # Odd rows move left; at the field's left edge (col 1) drop down,
            # except on the bottom row where we step into the spine.
            if c == 1:
                if r == H - 1:
                    return (r, 0)
                return (r + 1, c)
            return (r, c - 1)

    def _build(self):
        for r in range(self.height):
            for c in range(self.width):
                self._next[(r, c)] = self._successor(r, c)

    def next_cell(self, cell):
        return self._next.get(cell)

    def verify(self):
        """Return True iff the cycle visits every cell once and closes."""
        if not self.valid:
            return False
        start = (0, 0)
        seen = set()
        cur = start
        for _ in range(self.height * self.width):
            if cur in seen:
                return False
            seen.add(cur)
            nxt = self._next[cur]
            # Must be an orthogonal single step.
            if abs(nxt[0] - cur[0]) + abs(nxt[1] - cur[1]) != 1:
                return False
            cur = nxt
        return cur == start and len(seen) == self.height * self.width


class SafetyController:
    """Wraps a DQN preference vector with survival AND progress guarantees.

    Survival comes from the tail-reachability veto. Progress comes from a
    hunger-triggered fallback: if the network dawdles (goes too long without
    eating), the controller forces the shortest *safe* path to the food (A* for
    speed); if no safe path exists it follows the Hamiltonian cycle, which is
    guaranteed to reach the food eventually - so the snake can never loop forever.
    Control returns to the network as soon as the snake eats again.
    """

    def __init__(self, height=GRID_HEIGHT, width=GRID_WIDTH,
                 endgame_fill_ratio=0.5, use_hamiltonian=True, hunger_limit=None):
        self.height = height
        self.width = width
        self.area = height * width
        self.endgame_fill_ratio = endgame_fill_ratio
        self.cycle = HamiltonianCycle(height, width) if use_hamiltonian else None
        if self.cycle is not None and not self.cycle.valid:
            self.cycle = None

        # Cycle-index map: position -> its order in the Hamiltonian cycle (0..N-1).
        # This powers the guaranteed-win backbone (see choose_action_cycle).
        self.cycle_order = None   # list: index -> position
        self.cycle_index = None   # dict: position -> index
        if self.cycle is not None:
            order = []
            cur = (0, 0)
            for _ in range(self.area):
                order.append(cur)
                cur = self.cycle.next_cell(cur)
            self.cycle_order = order
            self.cycle_index = {pos: i for i, pos in enumerate(order)}

        # Backbone alignment bookkeeping (used by choose_action_cycle).
        self._cycle_prev_len = None

        # Progress / loop-breaking state.
        # Trigger "go get the food" mode after this many steps without eating.
        # One board diameter is a generous "you are clearly dawdling" threshold.
        self.hunger_limit = hunger_limit if hunger_limit is not None else (height + width)
        self.prev_len = None      # snake length last step (to detect eating / reset)
        self.hunger = 0           # steps since the snake last ate
        self.progress_mode = False  # currently forcing progress toward food?

    def reset(self):
        """Reset per-game progress state (safe to call at game start)."""
        self.prev_len = None
        self.hunger = 0
        self.progress_mode = False

    def _shortest_path_dir(self, snake_list, food):
        """BFS shortest path head->food over open cells; return first-step
        direction, or None if the food is currently unreachable.

        The tail cell is treated as walkable because it vacates as the snake
        moves. This is the 'A*-for-speed' component (plain BFS since the grid is
        unweighted, which gives the same shortest path more cheaply).
        """
        head = snake_list[0]
        if head == food:
            return None
        blocked = set(snake_list)
        blocked.discard(snake_list[-1])  # tail will move

        prev = {head: None}
        frontier = deque((head,))
        while frontier:
            cur = frontier.popleft()
            if cur == food:
                break
            r, c = cur
            for dr, dc in _ORTHO:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    nxt = (nr, nc)
                    if nxt not in prev and nxt not in blocked:
                        prev[nxt] = cur
                        frontier.append(nxt)
        if food not in prev:
            return None

        # Walk back from food to the first step off the head.
        step = food
        while prev[step] is not None and prev[step] != head:
            step = prev[step]
        if prev[step] is None:
            return None
        return (step[0] - head[0], step[1] - head[1])

    def evaluate_moves(self, game_engine):
        """Return per-action safety info without mutating the game.

        Returns a list of dicts (one per relative action 0/1/2) with keys:
            action, direction, alive, ate, reaches_tail, free_space, wins, is_cycle
        """
        snake_list = list(game_engine.snake)
        food = game_engine.food
        cur_dir = game_engine.direction
        head = snake_list[0]

        cycle_dir = None
        if self.cycle is not None:
            nxt = self.cycle.next_cell(head)
            if nxt is not None:
                cycle_dir = (nxt[0] - head[0], nxt[1] - head[1])

        results = []
        for action in range(3):
            direction = relative_to_absolute(cur_dir, action)
            new_body, ate, alive = simulate_move(
                snake_list, food, direction, self.height, self.width)

            info = {
                'action': action,
                'direction': direction,
                'alive': alive,
                'ate': ate,
                'reaches_tail': False,
                'free_space': 0,
                'wins': False,
                'is_cycle': (direction == cycle_dir),
            }

            if alive:
                if len(new_body) >= self.area:
                    # Filled the board on this move: a win.
                    info['wins'] = True
                    info['reaches_tail'] = True
                    info['free_space'] = self.area
                else:
                    reaches_tail, free_space = analyze_body(
                        new_body, self.height, self.width)
                    info['reaches_tail'] = reaches_tail
                    info['free_space'] = free_space

            results.append(info)
        return results

    def _update_hunger(self, snake_list):
        """Track steps-since-eating; reset on a new game or after eating."""
        length = len(snake_list)
        if self.prev_len is None or length < self.prev_len:
            # First call or a game reset (snake got shorter).
            self.hunger = 0
            self.progress_mode = False
        elif length > self.prev_len:
            # Ate this step -> hand control back to the network.
            self.hunger = 0
            self.progress_mode = False
        else:
            self.hunger += 1
        self.prev_len = length

        if self.hunger >= self.hunger_limit:
            self.progress_mode = True

    def choose_action(self, game_engine, prefs):
        """Pick a relative action (0/1/2).

        Args:
            game_engine: current game state.
            prefs: length-3 sequence of preferences (e.g. Q-values). Higher = better.
        """
        snake_list = list(game_engine.snake)
        self._update_hunger(snake_list)

        moves = self.evaluate_moves(game_engine)
        fill_ratio = len(snake_list) / self.area
        endgame = fill_ratio >= self.endgame_fill_ratio

        # An immediate winning move always wins.
        for m in moves:
            if m['wins']:
                return m['action']

        alive = [m for m in moves if m['alive']]
        safe = [m for m in alive if m['reaches_tail']]

        # In the endgame OR when the snake is dawdling (loop), we stop letting the
        # network wander and actively drive toward the food.
        want_progress = endgame or self.progress_mode

        if safe:
            if want_progress:
                # 1) A* for speed: take the shortest safe step toward the food.
                astar_dir = self._shortest_path_dir(snake_list, game_engine.food)
                if astar_dir is not None:
                    for m in safe:
                        if m['direction'] == astar_dir:
                            return m['action']
                # 2) No safe path to the food right now: in the endgame follow the
                #    Hamiltonian cycle (guaranteed to reach the food eventually,
                #    so no infinite loop); otherwise open up space so a path
                #    appears, rather than sitting in a tight loop.
                if endgame:
                    cyc = [m for m in safe if m['is_cycle']]
                    if cyc:
                        return max(cyc, key=lambda m: prefs[m['action']])['action']
                return max(safe, key=lambda m: (m['free_space'], prefs[m['action']]))['action']

            # Normal play: let the network drive among the safe moves.
            return max(safe, key=lambda m: (prefs[m['action']], m['free_space']))['action']

        if alive:
            # No move keeps a tail path -> survive as long as possible.
            # Prefer more open space, then a cycle move, then the network.
            key = lambda m: (m['free_space'], m['is_cycle'], prefs[m['action']])
            return max(alive, key=key)['action']

        # Every move is fatal (truly trapped) -> honour the network and accept fate.
        return max(range(3), key=lambda a: prefs[a])

    # ------------------------------------------------------------------
    # Guaranteed-win cycle backbone (Path A "closer")
    # ------------------------------------------------------------------
    def has_cycle_backbone(self):
        return self.cycle_index is not None

    def align_snake_to_cycle(self, game_engine):
        """Reposition the fresh snake so its body lies along the start of the
        Hamiltonian cycle. This is what makes cycle-following a *guaranteed* win:
        the safety proof only holds when the body is a contiguous arc of the cycle.

        Only safe to call on a fresh game (snake at its initial length).
        """
        if self.cycle_order is None:
            return False
        n = len(game_engine.snake)
        # Head at the highest index, tail at the lowest, so we advance forward.
        body = [self.cycle_order[i] for i in range(n - 1, -1, -1)]
        game_engine.snake = deque(body)
        head, neck = body[0], body[1]
        game_engine.direction = (head[0] - neck[0], head[1] - neck[1])
        game_engine.next_direction = game_engine.direction
        game_engine.food = game_engine.generate_food()
        self._cycle_prev_len = n   # mark aligned so auto-align won't re-fire
        return True

    def reset_cycle(self):
        self._cycle_prev_len = None

    def choose_action_cycle(self, game_engine, prefs=None, prefs_fn=None):
        """Guaranteed-win action selection: follow the Hamiltonian cycle as a
        backbone, letting the DQN take only shortcuts that CANNOT break the win.

        A shortcut (jump forward in cycle order) is allowed only if it:
          (1) does not overtake the tail in cycle order  (survival), and
          (2) moves strictly closer to the food IN CYCLE ORDER, i.e. without
              skipping past it (so the head always eventually lands on the food -
              no starvation loop).
        The DQN's preferences pick among the valid shortcuts; every option
        preserves the guarantee, so the network can only make the snake faster,
        never kill it.

        ``prefs`` may be a length-3 preference list, or omitted in favour of
        ``prefs_fn`` (a zero-arg callable returning such a list). ``prefs_fn`` is
        only invoked when there are >=2 valid shortcuts to disambiguate, so the
        expensive state/network evaluation is skipped on the vast majority of
        steps (which have a single forced move) - this keeps a full 900-cell win
        fast instead of tens of thousands of needless A*/network calls.

        Falls back to the survival shield if the cycle is unavailable (e.g. an odd
        grid height) or the snake is somehow off the cycle map.
        """
        if self.cycle_index is None:
            return self.choose_action(game_engine, prefs if prefs is not None else [0, 0, 0])

        # Auto-align at the start of a new game (snake shrank back to init length).
        length = len(game_engine.snake)
        if (self._cycle_prev_len is None or length < self._cycle_prev_len):
            if length == INITIAL_SNAKE_LENGTH:
                self.align_snake_to_cycle(game_engine)
        self._cycle_prev_len = len(game_engine.snake)

        snake = list(game_engine.snake)
        head, tail = snake[0], snake[-1]
        cur = game_engine.direction
        body_block = set(snake[:-1])          # tail vacates this step
        N = self.area

        idx = self.cycle_index
        if head not in idx or tail not in idx or game_engine.food not in idx:
            return self.choose_action(game_engine, prefs)

        head_i = idx[head]
        tail_i = idx[tail]
        food_i = idx[game_engine.food]
        d2t = (tail_i - head_i) % N            # cycle gap head->tail
        d2food_head = (food_i - head_i) % N    # cycle distance head->food

        natural_action = None
        shortcuts = []   # list of (action, d2n) that provably preserve the win

        for a in range(3):
            d = relative_to_absolute(cur, a)
            np_ = (head[0] + d[0], head[1] + d[1])
            if not (0 <= np_[0] < self.height and 0 <= np_[1] < self.width):
                continue
            if np_ in body_block:
                continue
            ni = idx[np_]
            d2n = (ni - head_i) % N            # cycle steps skipped by this move
            if d2n == 1:
                natural_action = a            # the plain cycle successor
            # Shortcut validity: don't overtake the tail, and advance toward the
            # food in cycle order without overshooting it.
            gap_ok = (d2t == 0) or (0 < d2n < d2t)
            if gap_ok and (food_i - ni) % N < d2food_head:
                shortcuts.append((a, d2n))

        if not shortcuts:
            if natural_action is not None:
                return natural_action
            # No natural step available (rare corner): fall back to survival shield.
            return self.choose_action(game_engine, [0, 0, 0] if prefs is None else prefs)

        if len(shortcuts) == 1:
            return shortcuts[0][0]

        # >=2 safe shortcuts. The biggest forward jump toward the food is the
        # speed-optimal choice (fewest steps to eat) and costs nothing to compute,
        # so by default we DON'T call the network here - that keeps a full 900-cell
        # win fast. The network is consulted only as a tie-break between equally-
        # good jumps, and only if a preference source was actually supplied.
        shortcuts.sort(key=lambda s: s[1], reverse=True)   # largest d2n first
        top = shortcuts[0][1]
        tied = [s for s in shortcuts if s[1] == top]
        if len(tied) == 1:
            return tied[0][0]
        if prefs is None and prefs_fn is not None:
            prefs = prefs_fn()
        if prefs is not None:
            return max(tied, key=lambda s: prefs[s[0]])[0]
        return tied[0][0]
