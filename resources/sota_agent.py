import jax
import jax.numpy as jnp
import pgx
from pgx.bridge_bidding import BridgeBidding

seed = 42
batch_size = 10
key = jax.random.PRNGKey(seed)


def act_randomly(rng_key, obs, mask):
    """Ignore observation and choose randomly from legal actions"""
    del obs
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(rng_key, logits=logits, axis=-1)


# Load the environment
env = BridgeBidding("dds_results/test_000.npy")
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

# Initialize the states
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, batch_size)
state = init_fn(keys)

# Run random simulation
while not (state.terminated | state.truncated).all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, state.observation, state.legal_action_mask)
    state = step_fn(state, action) 