import jax
import jax.numpy as jnp
import pgx
from pgx.experimental.utils import act_randomly
from pgx.bridge_bidding import BridgeBidding

seed = 42
batch_size = 10
key = jax.random.PRNGKey(seed)

# Prepare agent A and B
#   Agent A: random player
#   Agent B: baseline player provided by Pgx
A = 0
B = 1

# Load the environment
env = BridgeBidding()  # THE BRIDGE BIDDING CLASS
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

# Prepare baseline model
# Note that it additionally requires Haiku library ($ pip install dm-haiku)


# GO BOARD --------- (NOT FEASIBLE RIGHT NOW)
model_id = "go_9x9_v0"
model = pgx.make_baseline_model(model_id)

# Initialize the states
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, batch_size)
state = init_fn(keys)
print(f"Game index: {jnp.arange(batch_size)}")  #  [0 1 2 3 4 5 6 7 8 9]
print(f"Black player: {state.current_player}")  #  [1 1 0 1 0 0 1 1 1 1]
# In other words
print(f"A is black: {state.current_player == A}")  # [False False  True False  True  True False False False False]
print(f"B is black: {state.current_player == B}")  # [ True  True False  True False False  True  True  True  True]

# Run simulation
R = state.rewards
while not (state.terminated | state.truncated).all():
    # Action of random player A
    key, subkey = jax.random.split(key)
    action_A = jax.jit(act_randomly)(subkey, state)
    # Greedy action of baseline model B
    logits, value = model(state.observation)
    action_B = logits.argmax(axis=-1)

    action = jnp.where(state.current_player == A, action_A, action_B)
    state = step_fn(state, action)
    R += state.rewards

print(f"Return of agent A = {R[:, A]}")  # [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
print(f"Return of agent B = {R[:, B]}")  # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]