# Tic‑Tac‑Toe RL (Monte‑Carlo & Q‑Learning)

## A small Tic‑Tac‑Toe project that implements:

- First‑visit Monte‑Carlo policy evaluation for Q(s,a) under a random behavior policy, and

- Tabular Q‑learning with ε‑greedy exploration.

I wrote and tested this with Python 3.11.13 in a conda env, but it will run with any Python that supports type hints (Python 3.8+ recommended) and has numpy installed.

## Requirements

- Python (3.8+ recommended; tested on 3.11.13)

- numpy

All other imports are from the Python standard library.

Install numpy via pip:

`pip install numpy`

(Or use your package manager / conda environment if you prefer.)

## How to run

From the project root:

`python tictactoe.py`

The script prints progress and evaluation results to the console. Look in `__main__` for parameters you might want to tweak (episode counts, learning rate, epsilon schedule, seed).

## What the code implements (brief)

### MDP setup:

	- State: full board as a 9‑char string (row‑major) with X, O, ~ for empty.

	- Actions: indices 0–8 for empty cells.

	- Transition: X places deterministically; if game not over, O plays a uniformly random legal move (stochastic).

	- Reward: terminal only: +1 for X win, −1 for O win, 0 for draw. Episodes are episodic and undiscounted (γ = 1).

### Monte‑Carlo (first‑visit):


	- Episodes are generated using a random policy for X and a random opponent.

	- Because rewards are only terminal, every visited (s,a) in an episode receives the same final return G.

	- First‑visit logic: update Q(s,a) only on the first occurrence in the episode using incremental averaging:

Q ← Q + (1/n) (G - Q).

	- This is an on‑policy evaluation of the random policy. High variance per episode but unbiased with enough samples.

### Q‑learning:

	- Tabular, per‑step updates toward target r + γ max_a' Q(s',a').

	- Uses ε‑greedy action selection with random tie‑breaking.

	- Updates are online (per step) and off‑policy in nature (learns greedy values while exploring).


### Useful utilities in the code

    - mc_first_visit_Q(...) — compute first‑visit MC Q estimates under the random policy.

    - q_learning(...) — run Q‑learning and return learned Q.

    - show_policy_action(state, Q) — pretty‑print Q values on the 3×3 board and show the chosen best action.

    - compare_Qs_show(state, Q_mc, Q_q) — print MC vs Q‑learning Q values side by side for a given board.

## Notes & tips

    - Monte‑Carlo needs many episodes to reduce variance. Increase mc_episodes in `__main__` to improve MC estimates.

    - Q‑learning needs appropriate α, ε schedule and episodes for stable policy learning. Tweak alpha, eps_start, and decay_rate in q_learning(...).

    - The random seed is set for reproducibility — change or remove it if you want different random runs.

    - If you want to evaluate a different behavior policy for MC, replace random_policy in the MC functions.