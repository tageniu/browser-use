"""Example of using Browser-Use with AgentLab."""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_browser_use_agent():
	"""Run Browser-Use agent in AgentLab environment."""
	try:
		# Import AgentLab components
		from agentlab.experiments.study import make_study
		from browser_use.agentlab import BrowserUseAgent, BrowserUseAgentArgs
		from browser_use.agentlab.config import get_config_for_benchmark
		
		# Get configuration for the benchmark
		# Options: "webarena", "workarena", "miniwob", "gaia"
		agent_args = get_config_for_benchmark("webarena")
		
		# Create a study
		study = make_study(
			benchmark="webarena",
			agent_args=[agent_args],
			comment="Testing Browser-Use integration with AgentLab",
		)
		
		# Run the study with parallel execution
		logger.info("Starting AgentLab study with Browser-Use agent...")
		try:
			# Try to run with n_jobs parameter (newer versions)
			study.run(n_jobs=-1)  # Use all available cores
		except TypeError:
			# Fallback for older versions or different API
			try:
				# Check if study has a set_n_jobs method
				if hasattr(study, 'set_n_jobs'):
					study.set_n_jobs(-1)
					study.run()
				else:
					# Run without parallel execution
					logger.warning("Parallel execution not available, running sequentially")
					study.run()
			except Exception as e:
				logger.warning(f"Could not enable parallel execution: {e}")
				study.run()
		
		# Analyze results
		results = study.get_results()
		logger.info(f"Study completed. Results: {results}")
		
	except ImportError as e:
		logger.error(f"AgentLab not installed: {e}")
		logger.info("To use AgentLab integration, install it with: pip install agentlab")
		
		# Show example without AgentLab
		demo_without_agentlab()


def demo_without_agentlab():
	"""Demonstrate the agent without AgentLab installed."""
	from browser_use.agentlab import BrowserUseAgent, BrowserUseAgentArgs
	
	logger.info("\nDemonstrating Browser-Use AgentLab integration (without AgentLab):")
	
	# Create agent args
	agent_args = BrowserUseAgentArgs(
		model_name="gpt-4o",
		temperature=0.7,
		use_vision=True,
		headless=True,
		browser_type="chrome",
	)
	
	# Create agent
	agent = BrowserUseAgent(agent_args)
	
	# Create a mock observation (what BrowserGym would provide)
	mock_obs = {
		"dom_object": "<html><body><h1>Welcome</h1><button id='btn1'>Click me</button></body></html>",
		"url": "https://example.com",
		"title": "Example Page",
		"goal": "Click the button on the page",
		"screenshot": None,  # Would be base64 image data
		"tab_info": [{"id": 0, "url": "https://example.com", "title": "Example Page", "is_active": True}],
	}
	
	logger.info(f"Mock observation: {mock_obs}")
	
	# Get action (this would fail without proper LLM setup)
	try:
		action, info = agent.get_action(mock_obs)
		logger.info(f"Generated action: {action}")
		logger.info(f"Agent info: {info}")
	except Exception as e:
		logger.info(f"Expected error (no LLM configured): {e}")
		
		# Show what the action would look like
		logger.info("\nExample action that would be generated:")
		logger.info("Action: click(0)")
		logger.info("Info: {'think': 'I need to click the button', 'action_type': 'click'}")


def run_parallel_study():
	"""Example of running a study with explicit parallel configuration."""
	try:
		from agentlab.experiments.study import make_study
		from browser_use.agentlab import BrowserUseAgent, BrowserUseAgentArgs
		from browser_use.agentlab.config import get_config_for_benchmark
		import multiprocessing
		
		# Get number of available CPU cores
		n_cores = multiprocessing.cpu_count()
		logger.info(f"Available CPU cores: {n_cores}")
		
		# Get configuration
		agent_args = get_config_for_benchmark("miniwob")  # Use lighter benchmark for demo
		
		# Method 1: Try passing n_jobs to make_study
		try:
			study = make_study(
				benchmark="miniwob",
				agent_args=[agent_args],
				comment="Parallel execution test",
				n_jobs=n_cores - 1,  # Leave one core free
			)
			logger.info(f"Created study with n_jobs={n_cores - 1}")
		except TypeError:
			# Method 2: Create study without n_jobs
			study = make_study(
				benchmark="miniwob",
				agent_args=[agent_args],
				comment="Parallel execution test",
			)
			logger.info("Created study without n_jobs parameter")
		
		# Run the study
		logger.info("Running study...")
		study.run()
		
		# Get results
		results = study.get_results()
		logger.info(f"Study completed. Total tasks: {len(results)}")
		
	except ImportError as e:
		logger.error(f"AgentLab not installed: {e}")
	except Exception as e:
		logger.error(f"Error in parallel study: {e}")


def run_specific_task():
	"""Run a specific task with Browser-Use in AgentLab."""
	try:
		from agentlab.llm.llm_configs import OPENAI_GPT4O
		from agentlab.experiments import study_generators
		from browser_use.agentlab import BrowserUseAgent, BrowserUseAgentArgs
		
		# Configure the agent
		agent_args = BrowserUseAgentArgs(
			model_name="gpt-4o",
			temperature=0.5,
			use_vision=True,
			use_thinking=True,
		)
		
		# Create agent instance
		agent = BrowserUseAgent(agent_args)
		
		# Run on a specific WebArena task
		env = study_generators.get_benchmark_env("webarena", task_id="webarena.task_123")
		
		obs, info = env.reset()
		done = False
		step = 0
		
		while not done and step < 50:
			action, agent_info = agent.get_action(obs)
			obs, reward, done, info = env.step(action)
			logger.info(f"Step {step}: Action={action}, Reward={reward}, Done={done}")
			step += 1
			
		logger.info(f"Task completed. Final reward: {reward}")
		
	except Exception as e:
		logger.error(f"Error running specific task: {e}")


if __name__ == "__main__":
	# Try to run with AgentLab
	run_browser_use_agent()
	
	# Uncomment to run a parallel study
	# run_parallel_study()
	
	# Uncomment to run a specific task
	# run_specific_task()