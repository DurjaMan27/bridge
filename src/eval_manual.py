import jax
import os
import jax.numpy as jnp
import numpy as np
import distrax
import pickle
from src.duplicate import duplicate_step, Table_info
from src.models import make_forward_pass
from src.utils import single_play_step_two_policy_commpetitive_deterministic
from src.agent_client import make_http_agent_client
from src.callback_baseline import make_callback_baseline_agent, make_io_callback_baseline_agent
from baseline import BaselineAgent
import logging

logging.basicConfig(
  filename="src/outputs/debug_log.txt",
  level=logging.DEBUG,
  format="%(asctime)s [%(levelname)s] %(message)s"
)

ACTION_TO_STRING = {
  0: "Pass",
  1: "Double",
  2: "Redouble",
}

BID_LEVELS = ['1', '2', '3', '4', '5', '6', '7']
BID_SUITS = ['C', 'D', 'H', 'S', 'NT']
bid_index = 3
for level in BID_LEVELS:
    for suit in BID_SUITS:
        ACTION_TO_STRING[bid_index] = level + suit
        bid_index += 1

STRING_TO_ACTION = {v: k for k, v in ACTION_TO_STRING.items()}

def make_simple_duplicate_evaluate(
    eval_env,
    team1_activation,
    team1_model_type,
    team2_activation,
    team2_model_type,
    num_eval_envs,
    team1_server_url = None,
    team2_server_url = None,
):

    if team1_model_type == "baseline":
        if team1_server_url:
            team1_agent_fn = make_callback_baseline_agent()
        else:
            team1_forward_pass = None
    else:
        team1_forward_pass = make_forward_pass(
            activation=team1_activation,
            model_type=team1_model_type,
        )

    if team2_model_type == "baseline":
        if team1_server_url:
            team2_agent_fn = make_callback_baseline_agent()
    else:
        team2_forward_pass = make_forward_pass(
            activation=team2_activation,
            model_type=team2_model_type,
        )

    def duplicate_evaluate(
        team1_params,
        team2_params,
        rng_key
    ):

        step_fn = duplicate_step(eval_env.step)
        rng_key, sub_key = jax.random.split(rng_key)
        subkeys = jax.random.split(sub_key, num_eval_envs)
        state = jax.vmap(eval_env.init)(subkeys)
        cum_return = jnp.zeros(num_eval_envs)

        # state_decode goes here

        table_a_info = Table_info(
            terminated=state.terminated,
            rewards=state.rewards,
            last_bid=state._last_bid,
            last_bidder=state._last_bidder,
            call_x=state._call_x,
            call_xx=state._call_xx,
        )
        table_b_info = Table_info(
            terminated=state.terminated,
            rewards=state.rewards,
            last_bid=state._last_bid,
            last_bidder=state._last_bidder,
            call_x=state._call_x,
            call_xx=state._call_xx,
        )

        jax.debug.print("=== INITIAL STATE DEBUG ===")
        jax.debug.print("Initial state terminated: {}", state.terminated)
        jax.debug.print("Initial state rewards: {}", state.rewards)
        jax.debug.print("Initial table A terminated: {}", table_a_info.terminated)
        jax.debug.print("Initial table B terminated: {}", table_b_info.terminated)

        cum_return = jnp.zeros(num_eval_envs)
        count = 0

        def get_fn(x, i):
            return x[i]
        
        def cond_fn(tup):
            (state, _, _, _, _, _) = tup
            return ~state.terminated.all()

        def actor_make_action(state):

            if team1_model_type == "baseline" and team1_server_url:
                action, pi_probs = team1_agent_fn(state)

                # jax.debug.print("=== BASELINE AGENT TURN ===")
                # jax.debug.print("Current player: {}", state.current_player)
                # jax.debug.print("Legal actions: {}", state.legal_action_mask)
                # jax.debug.print("Last bid: {}", state._last_bid)
                # jax.debug.print("Last bidder: {}", state._last_bidder)

                # if action < 0 or action >= 38:
                #     jax.debug.print("ERROR: Invalid action index: {}", action)
                # elif not state.legal_action_mask[action]:
                #     jax.debug.print("ERROR: Action {} not legal", action)

                return action, pi_probs
            elif team1_model_type == "baseline":

                jax.debug.print("=== BASELINE AGENT TURN ===")
                jax.debug.print("Current player: {}", state.current_player)
                jax.debug.print("Legal actions: {}", state.legal_action_mask)
                jax.debug.print("Last bid: {}", state._last_bid)
                jax.debug.print("Last bidder: {}", state._last_bidder)

                # legal_mask = state.legal_action_mask[0]

                # action_scores = jnp.where(legal_mask, jnp.arange(38), -jnp.inf)
                # action = jnp.argmax(action_scores)

                # action = jnp.where(legal_mask, action, 0)

                baseline_agent = BaselineAgent()
                action = baseline_agent.make_bid(state)

                if not isinstance(int, action):
                    print("\n\n\nALERT =======")
                    print("Returned", action, "\n\n\n\n")
                    action = 0

                jax.debug.print("Baseline agent selected action: {}", action)

                pi_probs = jnp.zeros(state.legal_action_mask.shape)
                pi_probs = pi_probs.at[action].set(1.0)
                return action, pi_probs
            else:
                logits, value = team1_forward_pass.apply(
                    team1_params, state.observation
                )

                masked_logits = logits + jnp.finfo(np.float64).min * (
                    ~state.legal_action_mask,
                )
                masked_pi = distrax.Categorical(logits=masked_logits)
                pi = distrax.Categorical(logits=logits)
                return (masked_pi.mode(), pi.probs)
            
        def opp_make_action(state):
            if team2_model_type == "baseline" and team2_server_url:
                return team2_agent_fn(state)
            elif team2_model_type == "baseline":
                # legal_mask = state.legal_action_mask[0]

                baseline_agent = BaselineAgent()
                action = baseline_agent.make_bid(state)

                if not isinstance(int, action):
                    print("\n\n\nALERT =======")
                    print("Returned", action, "\n\n\n\n")
                    action = 0

                pi_probs = jnp.zeros(state.legal_action_mask.shape)
                pi_probs = pi_probs.at[action].set(1.0)
                return action, pi_probs
            else:
                logits, value = team2_forward_pass.apply(
                    team2_params, state.observation
                )
                masked_logits = logits + jnp.finfo(np.float64).min * (
                    ~state.legal_action_mask
                )
                masked_pi = distrax.Categorical(logits=masked_logits)
                pi = distrax.Categorical(logits=logits)
                return (masked_pi.mode(), pi.probs)
            
        def make_action(state):
            return jax.lax.cond(
                (state.current_player == 0) | (state.current_player == 1),
                lambda: actor_make_action(state),
                lambda: opp_make_action(state),
            )
        
        def loop_fn(tup):

            (
                state,
                table_a_info,
                table_b_info,
                cum_return,
                rng_key,
                count,
            ) = tup


            jax.debug.print("=== STEP {} DEBUG ===", count)
            jax.debug.print("State terminated: {}", state.terminated)
            jax.debug.print("State rewards: {}", state.rewards)
            jax.debug.print("Current player: {}", state.current_player)
            jax.debug.print("Cumulative return: {}", cum_return)
            jax.debug.print("Last bid: {}", state._last_bid)
            jax.debug.print("Bidding history length: {}", jnp.sum(state._bidding_history != -1))

            (action, pi_probs) = jax.vmap(make_action)(state)
            rng_key, _rng = jax.random.split(rng_key)
            (state, table_a_info, table_b_info) = jax.vmap(step_fn)(
                state, action, table_a_info, table_b_info
            )

            jax.debug.print("BEFORE duplicate logic:")
            jax.debug.print("  state.terminated: {}", state.terminated)
            jax.debug.print("  table_a_info.terminated: {}", table_a_info.terminated)
            jax.debug.print("  table_b_info.terminated: {}", table_b_info.terminated)
            jax.debug.print("  state.rewards: {}", state.rewards)
            jax.debug.print("  table_a_info.rewards: {}", table_a_info.rewards)

            duplicate_comparison = table_a_info.terminated & state.terminated & table_b_info.terminated
            jax.debug.print("After step - State terminated: {}", state.terminated)
            jax.debug.print("After step - State rewards: {}", state.rewards)
            jax.debug.print("After step - Table A terminated: {}", table_a_info.terminated)
            jax.debug.print("After step - Table B terminated: {}", table_b_info.terminated)
            jax.debug.print("Duplicate comparison check: {}", duplicate_comparison)
            jax.debug.print("Any duplication comparison: {}", jnp.any(duplicate_comparison))

            jax.debug.print("After duplication logic - State rewards: {}", state.rewards)

            cum_return = cum_return + jax.vmap(get_fn)(
                state.rewards,
                jnp.zeros_like(state.current_player)
            )

            jax.debug.print("Updated cumulative return: {}", cum_return)

            count += 1
            return (
                state,
                table_a_info,
                table_b_info,
                cum_return,
                rng_key,
                count,
            )
        
        (
            state,
            table_a_info,
            table_b_info,
            cum_return,
            _,
            count,
        ) = jax.lax.while_loop(
            cond_fn,
            loop_fn,
            (
                state,
                table_a_info,
                table_b_info,
                cum_return,
                rng_key,
                count,

            ),
        )
        std_error = jnp.std(cum_return, ddof=1) / jnp.sqrt(len(cum_return))
        win_rate = jnp.sum(cum_return > 0) / num_eval_envs
        log_info = (cum_return.mean(), std_error, win_rate)
        return log_info, table_a_info, table_b_info
    
    return duplicate_evaluate