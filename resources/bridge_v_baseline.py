import jax, os
import jax.numpy as jnp
from dotenv import load_dotenv
from pgx.experimental.utils import act_randomly
from pgx.bridge_bidding import BridgeBidding
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

# Machine Learning MODEL:
MODEL = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

def format_prompt_with_state(current_player, observation, legal_action_mask):
    """
    current_player: list or np.array of ints, shape (batch_size,)
    observation: list of lists or 2D np.array of bools, shape (batch_size, obs_dim)
    legal_action_mask: list of lists or 2D np.array of bools, shape (batch_size, num_actions)
    """
    prompt = (
        "You are an AI agent selecting exactly one valid action per batch element.\n"
        "For each batch element, you will be given:\n"
        "- The current player (an integer id)\n"
        "- The observation vector (a list of booleans)\n"
        "- The legal action mask (a list of booleans indicating which actions are valid)\n\n"
        "For each batch element, select exactly one valid action index from those where legal_action_mask is True.\n"
        "Output ONLY a JSON list of integers representing the chosen action indices, one per batch element.\n"
        "Do not include any extra commentary or explanation.\n\n"
        "Batch elements:\n"
    )
    for i, (player, obs, mask) in enumerate(zip(current_player, observation, legal_action_mask), 1):
        valid_actions = [idx for idx, val in enumerate(mask) if val]
        prompt += (
            f"{i}:\n"
            f"  Current player: {player}\n"
            f"  Observation: {obs}\n"
            f"  Legal actions mask: {mask}\n"
            f"  Valid action indices: {valid_actions}\n\n"
        )
    prompt += "Choose valid actions accordingly."
    return prompt

def query(MODEL: ChatOpenAI, current_player, obs_batch, mask_batch):
    messages = [
      SystemMessage(
          content="You are a world champion computer bridge player, capable of discerning vectors of Bridge information to make the best possible decision to win the game."
      ),
      HumanMessage(
          content=format_prompt_with_state(current_player, obs_batch, mask_batch)
      )
    ]

    response = MODEL.invoke(messages)

    return response.content

# Setup
seed = 42
batch_size = 10
key = jax.random.PRNGKey(seed)

# Define player indices
A = 0  # Random player
B = 1  # Placeholder model

# Load the Bridge Bidding environment
env = BridgeBidding()
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

# Initialize game states
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, batch_size)
state = init_fn(keys)

print(f"Game index: {jnp.arange(batch_size)}")
print(f"Current player: {state.current_player}")
print(f"A's turn: {state.current_player == A}")
print(f"B's turn: {state.current_player == B}")

def bridge_bidding_placeholder_model(state, player):
    obs_batch = state.observation           # Shape: (batch_size, 480)
    mask_batch = state.legal_action_mask    # Shape: (batch_size, 38)

    actions = query(MODEL, player, obs_batch, mask_batch)

    return actions

# Run simulation
R = state.rewards
while not (state.terminated | state.truncated).all():
    # Random action for player A
    key, subkey = jax.random.split(key)
    action_A = jax.jit(act_randomly)(subkey, state)

    # Placeholder model action for player B
    action_B = bridge_bidding_placeholder_model(state.current_player, state)

    # Choose action based on current player
    action = jnp.where(state.current_player == A, action_A, action_B)
    state = step_fn(state, action)
    R += state.rewards

# Output results
print(f"Return of agent A = {R[:, A]}")
print(f"Return of agent B = {R[:, B]}")
