## Important files in directory:
- **run_bidding.py** --> entry point to framework, used to test NNs, baseline agent, and LLM
- **baseline.py** --> logic for baseline if-else bidding agent
- **llm.py** --> logic for the LLM bidding agent, including prompt generation and OpenAI API call
- **agent_server.py** --> hosts the local server responsible for directing "make_bid" calls for both baseline and LLM agents
- **src/eval_manual.py** --> Uses JAX logic to pass state variables into the non-JAX bidding agents
- **src/callback_baseline.py** --> Uses JAX's pure_callback to pass JAX state variable into baseline bidding agent; called by eval_manual.py
- **src/callback_llm.py** --> Uses JAX's pure_callback to pass JAX state variable into LLM bidding agent; called by eval_manual.py
- **src/*** --> all other files from the PGX bridge bidding repo, used for the original NN vs. NN testing framework

## To RUN (use virtual env):
### Execution Mode:
- The execution now requires specifying the agents for Team 1 (NS) and Team 2 (EW). The --team1_agent and --team2_agent arguments accept either baseline or llm.
- Local Mode (Agent logic runs directly in Python via JAX callback):
- To run a Baseline Agent (T1) vs. LLM Agent (T2) without the FastAPI server:
`python run_bidding.py --team1_agent baseline --team2_agent llm`

### Server Mode (Agent logic runs via HTTP requests to agent_server.py):

- This is typically used for external agents (like the LLM) to ensure clean separation, but can be used for any agent type.
- To run an LLM Agent (T1) vs. Baseline Agent (T2) using the server:
`python run_bidding.py --use_server --team1_agent llm --team2_agent baseline`

## Most recent update:
- tested llm against baseline and stored results in **/output_results**
- tested baseline against itself and stored results in **/output_results**
- updated and improved logic of baseline agent, much more predictable decisions
- created a more detailed prompt for LLM to perform better

## To-Do:
- test baseline against NN
- test LLM against NN