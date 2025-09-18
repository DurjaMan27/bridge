import jax
import os
import jax.numpy as jnp
import numpy as np
import distrax
import pickle
from src.duplicate import duplicate_step, Table_info
from src.models import make_forward_pass
from src.utils import single_play_step_two_policy_commpetitive_deterministic
import logging

logging.basicConfig(
    filename="src/outputs/debug_log.txt",   # file to write to
    level=logging.DEBUG,        # log everything
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def make_simple_duplicate_evaluate(
    eval_env,
    team1_activation,
    team1_model_type,
    team2_activation,
    team2_model_type,
    num_eval_envs,
):

    first_time_1 = False
    first_time_2 = False
    first_time_3 = False

    # CHANGE 1: allow for either NN forward pass OR a rule-based model
    # If model_type == "baseline", instead of building a forward_pass,
    # we will expect an object with a .make_bid(state) method.
    if team1_model_type == "baseline":
        team1_forward_pass = None  # placeholder, not used
    else:
        team1_forward_pass = make_forward_pass(
            activation=team1_activation,
            model_type=team1_model_type,
        )

    if team2_model_type == "baseline":
        team2_forward_pass = None
    else:
        team2_forward_pass = make_forward_pass(
            activation=team2_activation,
            model_type=team2_model_type,
        )

    def duplicate_evaluate(
        team1_params,
        team2_params,
        rng_key,
    ):
        step_fn = duplicate_step(eval_env.step)
        rng_key, sub_key = jax.random.split(rng_key)
        subkeys = jax.random.split(sub_key, num_eval_envs)
        state = jax.vmap(eval_env.init)(subkeys)
        cum_return = jnp.zeros(num_eval_envs)
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

        cum_return = jnp.zeros(num_eval_envs)
        count = 0

        def get_fn(x, i):
            return x[i]

        def cond_fn(tup):
            (state, _, _, _, _, _) = tup
            return ~state.terminated.all()

        # ⚠️ CHANGE 2: add branches for baseline policy
        def actor_make_action(state):

            if team1_model_type == "baseline":
                if not first_time_1:
                    print(state.observation)
                    logging.debug(f"STATE OBSERVATION: \n{state.observation}")
                    first_time_1 = True
                # expect team1_params to actually be a baseline model object
                action = team1_params.make_bid(state)
                # return dummy probs (e.g., one-hot)
                pi_probs = jnp.zeros(state.legal_action_mask.shape)
                pi_probs = pi_probs.at[action].set(1.0)
                return action, pi_probs
            else:
                logits, value = team1_forward_pass.apply(
                    team1_params, state.observation
                )

                if not first_time_3:
                    print("LOGITS SHAPE:", logits.shape)
                    print("LEGAL MASK:", state.legal_action_mask)
                    logging.debug(f"LOGITS SHAPE: \n{logits.shape}")
                    logging.debug(f"LEGAL MASK: \n{state.legal_action_mask}")
                    first_time_3 = True

                masked_logits = logits + jnp.finfo(np.float64).min * (
                    ~state.legal_action_mask
                )
                masked_pi = distrax.Categorical(logits=masked_logits)
                pi = distrax.Categorical(logits=logits)
                return (masked_pi.mode(), pi.probs)

        def opp_make_action(state):
            if team2_model_type == "baseline":
                action = team2_params.make_bid(state)
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
            
            if not first_time_2:
                print("STATE STRUCTURE:", state)
                logging.debug(f"SECOND TIME STATE: \n{state}")
                first_time_2 = True

            (action, pi_probs) = jax.vmap(make_action)(state)
            rng_key, _rng = jax.random.split(rng_key)
            (state, table_a_info, table_b_info) = jax.vmap(step_fn)(
                state, action, table_a_info, table_b_info
            )
            cum_return = cum_return + jax.vmap(get_fn)(
                state.rewards, jnp.zeros_like(state.current_player)
            )
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
