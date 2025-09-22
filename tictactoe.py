import random
from collections import defaultdict
import numpy as np
import math
import time

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

X, O, E = "X", "O", "~"  # players and empty


def empty_board():
    return tuple([E] * 9)  # immutable state


WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # cols
    (0, 4, 8),
    (2, 4, 6),  # diagonals
]


def check_winner(state):
    for a, b, c in WIN_LINES:
        line = (state[a], state[b], state[c])
        if line == (X, X, X):
            return X
        if line == (O, O, O):
            return O
    if E not in state:
        return "DRAW"
    return None


def legal_actions(state):
    return [i for i, v in enumerate(state) if v == E]


def place(state, idx, p):
    s = list(state)
    s[idx] = p
    return tuple(s)


def opponent_move(state):
    """Opponent O plays uniformly random legal move. If terminal, returns state."""
    if check_winner(state):
        return state
    acts = legal_actions(state)
    if not acts:
        return state
    return place(state, random.choice(acts), O)


def step(state, action):
    """Agent X acts; environment responds with O's random move.
    Returns (next_state, reward, done) from X's perspective.
    """
    assert action in legal_actions(state), "Illegal action"
    s1 = place(state, action, X)
    w = check_winner(s1)
    if w == X:
        return s1, +1.0, True
    if w == O:
        return s1, -1.0, True
    if w == "DRAW":
        return s1, 0.0, True

    # Opponent acts
    s2 = opponent_move(s1)
    w2 = check_winner(s2)
    if w2 == X:
        return s2, +1.0, True
    if w2 == O:
        return s2, -1.0, True
    if w2 == "DRAW":
        return s2, 0.0, True
    return s2, 0.0, False  # ongoing


def random_policy(state):
    acts = legal_actions(state)
    return random.choice(acts)


# ---------- Monte Carlo Policy Evaluation (First-Visit) ----------
def generate_episode(policy):
    s = empty_board()
    states: list[str] = [get_state_str(s)]
    r = 0.0

    done = False
    while not done:
        a = policy(s)
        s, r, done = step(s, a)
        state_str = get_state_str(s)
        states.append(state_str)

    return states, r


def mc_first_visit_V(num_episodes=50000):
    V = defaultdict(float)
    returns_count = defaultdict(int)
    my_policy = random_policy

    for _ in range(num_episodes):
        states, r = generate_episode(my_policy)
        seen_states = set()

        for state in states:
            if state not in seen_states:
                returns_count[state] += 1  # increment first
                V[state] += (r - V[state]) / returns_count[state]
                seen_states.add(state)

    return V


# ---------- Monte Carlo Policy Evaluation (First-Visit) for Q ----------
def generate_episode_2(policy):
    s = empty_board()
    states: list[str] = [get_state_str(s)]
    actions: list[int] = []
    r = 0.0

    done = False
    while not done:
        a = policy(s)
        s, r, done = step(s, a)
        state_str = get_state_str(s)
        states.append(state_str)
        actions.append(a)

    return states, r, actions


def mc_first_visit_Q(num_episodes=50000, num_actions=9):
    """First-visit MC estimate of Q for the given policy (random by default).
    Uses incremental averaging (online) for each state-action first visit.
    """
    Q = defaultdict(lambda: [0.0] * num_actions)
    returns_count = defaultdict(lambda: [0] * num_actions)
    my_policy = random_policy  # your policy function

    for _ in range(num_episodes):
        states, r, actions = generate_episode_2(my_policy)
        seen_state_actions = set()

        # Align states and actions: states[i] is the state BEFORE taking
        # actions[i] (we appended initial state and then terminal state).
        for idx in range(len(actions)):
            state = states[idx]
            action = actions[idx]

            if (state, action) not in seen_state_actions:
                returns_count[state][action] += 1
                n = returns_count[state][action]
                # incremental average:
                Q[state][action] += (r - Q[state][action]) / n
                seen_state_actions.add((state, action))

    # Note: values are already incremental averages; no additional division.
    return Q


# ---------- Pretty-print helpers and greedy helpers ----------
def printValueStates(stateValues: list[float]):
    print("\n")
    width = 6  # enough for numbers like -0.2857
    for row in range(3):
        row_values = stateValues[row * 3 : (row + 1) * 3]
        print(" | ".join(f"{v:{width}.2f}" for v in row_values))
        if row < 2:
            print("-" * (width * 3 + 6))


def printValueStatesWithMoves(stateValues: list[float], board: list[str]):
    print("\n")
    width = 6  # enough for numbers
    for row in range(3):
        row_items = []
        for col in range(3):
            idx = row * 3 + col
            if board[idx] == E:
                row_items.append(f"{stateValues[idx]:{width}.2f}")
            else:
                row_items.append(f"{board[idx]:^{width}}")  # center X or O
        print(" | ".join(row_items))
        if row < 2:
            print("-" * (width * 3 + 6))


def greedy_one_step_with_Q(state, Q):
    """Print Q-values (for legal moves) on the board and return best action and value.
    Q is expected to be dict: state_str -> list[num_actions].
    """
    state_string = get_state_str(state)
    state_legal_actions = legal_actions(state)
    state_values: list[float] = Q[state_string]  # action-values list (default 0s)

    # Max value over only legal actions (random tie-break)
    best_action = argmax_random(state_legal_actions, lambda a: state_values[a])
    max_value = state_values[best_action]

    # Report Out
    printValueStatesWithMoves(state_values, state)
    print(f"Best Action: {best_action} | Action Value: {max_value:.2f}")

    return best_action, max_value


def show_policy_action(state, Q):
    """Helper to print Q-grid and chosen action for a given (list|tuple) board."""
    return greedy_one_step_with_Q(state, Q)


def improved_policy_from_Q(Q):
    """Return greedy policy from Q with random tie-breaking."""
    def pi(s):
        s_str = get_state_str(s)
        legal = legal_actions(s)
        return argmax_random(legal, lambda a: Q[s_str][a])
    return pi


# ---------- Step By Step Q-Learning --------------
def argmax_random(choices, key_fn):
    """Return an element of choices that maximizes key_fn, tie-broken randomly."""
    best_val = -math.inf
    best_items = []
    for c in choices:
        v = key_fn(c)
        if v > best_val + 1e-12:
            best_val = v
            best_items = [c]
        elif abs(v - best_val) <= 1e-12:
            best_items.append(c)
    return random.choice(best_items)


def q_learning(
    num_episodes=50000,
    alpha=0.5,
    gamma=1.0,
    eps_start=0.2,
    eps_end=0.01,
    decay_rate=0.99995,
    num_actions=9,
    eval_every=5000,
    eval_n=2000,
):
    Q = defaultdict(lambda: [0.0] * num_actions)
    eps = eps_start

    for ep in range(1, num_episodes + 1):
        s = empty_board()
        done = False

        while not done:
            s_str = get_state_str(s)
            legal = legal_actions(s)

            # ε-greedy (with random tie-break)
            if random.random() < eps:
                a = random.choice(legal)
            else:
                a = argmax_random(legal, lambda x: Q[s_str][x])

            s2, r, done = step(s, a)
            s2_str = get_state_str(s2)

            if done:
                target = r
            else:
                # X to act at s2 (non-terminal) -> look at legal actions there
                next_legal = legal_actions(s2)
                if next_legal:
                    # random tie-break when selecting best next action
                    best_next_action = argmax_random(
                        next_legal, lambda a2: Q[s2_str][a2]
                    )
                    best_next = Q[s2_str][best_next_action]
                else:
                    best_next = 0.0
                target = r + gamma * best_next

            Q[s_str][a] += alpha * (target - Q[s_str][a])
            s = s2

        # decay epsilon
        eps = max(eps_end, eps * decay_rate)

        # optional: periodic quick evaluation to watch progress
        if eval_every and (ep % eval_every == 0):
            pol = q_policy(Q)
            w, d, l = play_many(pol, n=eval_n)
            print(
                f"Episode {ep:6d} eps={eps:.4f} eval win/draw/loss = "
                f"{w:.3f}/{d:.3f}/{l:.3f}"
            )

    return Q


def q_policy(Q):
    """Return greedy policy from Q with random tie-breaking."""
    def policy(state):
        s_str = get_state_str(state)
        legal = legal_actions(state)
        return argmax_random(legal, lambda x: Q[s_str][x])
    return policy


# ---------- Evaluation ----------
def play_many(policy, n=10000):
    wins = draws = losses = 0
    for _ in range(n):
        s = empty_board()
        done = False
        while not done:
            a = policy(s)
            s, r, done = step(s, a)
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            draws += 1
    return wins / n, draws / n, losses / n


# ---- Helper Fuctions -----
def get_state_str(state):
    return "".join(state)


# ---------- New comparison helpers ----------
def compare_Qs_show(state, Q_mc, Q_q, label_mc="Monte-Carlo Q", label_q="Q-learning Q"):
    print(f"\n===== {label_mc} =====")
    greedy_one_step_with_Q(state, Q_mc)
    print(f"\n===== {label_q} =====")
    greedy_one_step_with_Q(state, Q_q)
    print("\n-----------------------------")


if __name__ == "__main__":
    # quick random-policy V estimate
    print("Estimating V under random policy...")
    V = mc_first_visit_V(50000)
    v_empty = V.get(get_state_str(empty_board()), 0.0)
    print("V_random(empty) ≈", round(v_empty, 4))

    print("Evaluating random vs random:")
    w, d, l = play_many(random_policy, 10000)
    print(f"Random policy as X vs random O: win={w:.3f}, draw={d:.3f}, loss={l:.3f}\n")

    # ---------- Compute Monte-Carlo Q and Q-learning Q ----------
    # WARNING: increasing these will improve estimates but increase runtime.
    mc_episodes = 200000      # adjust as needed
    q_episodes = 200000       # adjust as needed

    t0 = time.time()
    print(f"Computing Monte-Carlo Q with {mc_episodes} episodes (random policy)...")
    mcQ = mc_first_visit_Q(num_episodes=mc_episodes)
    print(f"Done MC in {time.time()-t0:.1f}s\n")

    t0 = time.time()
    print(f"Running Q-learning for {q_episodes} episodes...")
    qQ = q_learning(num_episodes=q_episodes, alpha=0.5, eps_start=0.6)
    print(f"Done Q-learning in {time.time()-t0:.1f}s\n")

    # Evaluate learned greedy Q policy
    policy = q_policy(qQ)
    w_q, d_q, l_q = play_many(policy, 10000)
    print("Evaluating q-learning (greedy) vs random:")
    print(
        f"q-learning (as X) vs random O: win={w_q:.3f}, draw={d_q:.3f}, loss={l_q:.3f}\n"
    )

    # Test states to compare values
    test_state = ["X", "O", "~", "X", "O", "~", "~", "~", "~"]
    test_state_2 = ["X", "X", "~", "~", "~", "~", "O", "O", "~"]
    test_state_3 = ["X", "~", "~", "~", "~", "~", "O", "~", "~"]
    test_state_4 = ["X", "~", "~", "X", "~", "~", "O", "O", "~"]
    test_state_5 = ["X", "~", "X", "X", "~", "O", "O", "O", "~"]

    # Print side-by-side MC vs Q-learning for each test state
    compare_Qs_show(test_state, mcQ, qQ, label_mc="MC (random policy)", label_q="Q-learning (learned)")
    compare_Qs_show(test_state_2, mcQ, qQ, label_mc="MC (random policy)", label_q="Q-learning (learned)")
    compare_Qs_show(test_state_3, mcQ, qQ, label_mc="MC (random policy)", label_q="Q-learning (learned)")
    compare_Qs_show(test_state_4, mcQ, qQ, label_mc="MC (random policy)", label_q="Q-learning (learned)")
    compare_Qs_show(test_state_5, mcQ, qQ, label_mc="MC (random policy)", label_q="Q-learning (learned)")