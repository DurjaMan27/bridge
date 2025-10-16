## Important files in directory:
- **test_baseline_debug.py** --> entry point to framework, used to test NNs, baseline agent, and LLM
- **baseline.py** --> logic for baseline if-else bidding agent
- **agent_server.py** --> hosts the local server responsible for directing "make_bid" calls if server is to be used
- **src/eval_manual.py** --> Uses JAX logic to pass state variables into the non-JAX bidding agents
- **src/callback_baseline.py** --> Uses JAX's pure_callback to pass JAX state variable into bidding agent; called by eval_manual.py
- **src/\*** --> all other files from the PGX bridge bidding repo, used for the original NN vs. NN testing framework

## To RUN (use virtual env):
### With server:
python test_baseline_debug.py --use_server
### Without server:
python test_baseline_debug.py

## Most recent update:
- server bottleneck still exists -- stopping at 1074 bids for some reason
- added more error logging and helper function
- added batch function to split up calls to eval_env for large env numbers

## To-Do:
- explore whether the eval_envs test the same card hands every time - THEY DO (RNG KEY)
- explore which card hands are "equal" to create a more reproducible testing set
- add LLM agent
- improve baseline agent
- test baseline against NN
- test LLM against baseline, NN