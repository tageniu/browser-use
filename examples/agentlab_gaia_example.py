"""Example of using Browser-Use with AgentLab on GAIA benchmark."""

import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_gaia_benchmark():
	"""Run Browser-Use agent on GAIA benchmark."""
	try:
		# Import AgentLab components
		from agentlab.experiments.study import make_study
		from browser_use.agentlab import BrowserUseAgent, BrowserUseAgentArgs
		from browser_use.agentlab.config import get_config_for_benchmark
		
		# Configure for GAIA benchmark
		agent_args = BrowserUseAgentArgs(
			model_name="gpt-4o",
			temperature=0.5,
			use_vision=True,
			use_thinking=True,
			use_accessibility_tree=True,
			max_actions_per_step=3,  # GAIA tasks may require multiple actions
			planner_interval=3,  # More frequent planning for complex tasks
			headless=True,
			viewport_width=1280,
			viewport_height=960,
		)
		
		# Set reproducibility mode for consistent results
		agent_args.set_reproducibility_mode()
		
		# Create a study for GAIA benchmark
		study = make_study(
			benchmark="gaia",  # GAIA benchmark
			agent_args=[agent_args],
			comment="Testing Browser-Use on GAIA benchmark",
		)
		
		# Run the study with parallel execution
		logger.info("Starting GAIA benchmark study with Browser-Use agent...")
		logger.info("This will run a subset of GAIA tasks to test functionality")
		
		try:
			# Try to run with parallel execution
			study.run(n_jobs=2)  # Run 2 tasks in parallel for testing
		except TypeError:
			# Fallback to sequential execution
			logger.info("Running sequentially...")
			study.run()
		
		# Analyze results
		results = study.get_results()
		logger.info(f"Study completed. Total tasks: {len(results)}")
		
		# Display results summary
		if results:
			success_count = sum(1 for r in results if r.get('success', False))
			logger.info(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
			
			# Show first few results
			for i, result in enumerate(results[:3]):
				logger.info(f"\nTask {i+1}:")
				logger.info(f"  Task ID: {result.get('task_id', 'N/A')}")
				logger.info(f"  Success: {result.get('success', False)}")
				logger.info(f"  Steps taken: {result.get('steps', 'N/A')}")
		
		return results
		
	except ImportError as e:
		logger.error(f"AgentLab not installed: {e}")
		logger.info("To use AgentLab integration, install it with: pip install agentlab")
		
		# Show mock GAIA task example
		demo_gaia_task()
		return None
	except Exception as e:
		logger.error(f"Error running GAIA benchmark: {e}")
		return None


def demo_gaia_task():
	"""Demonstrate a GAIA-style task without AgentLab."""
	logger.info("\nDemonstrating GAIA-style task (mock):")
	logger.info("GAIA tasks typically involve:")
	logger.info("- Web search and information gathering")
	logger.info("- Multi-step reasoning")
	logger.info("- File manipulation and analysis")
	logger.info("- Complex question answering")
	
	from browser_use.agentlab import BrowserUseAgent, BrowserUseAgentArgs
	
	# Create agent configured for GAIA
	agent_args = BrowserUseAgentArgs(
		model_name="gpt-4o",
		temperature=0.3,  # Lower temperature for accuracy
		use_vision=True,
		use_thinking=True,
		max_actions_per_step=3,
		enable_memory=True,  # Important for multi-step tasks
	)
	
	# Create agent
	agent = BrowserUseAgent(agent_args)
	
	# Example GAIA-style observation
	mock_obs = {
		"dom_object": """
		<html>
		<body>
			<h1>Research Portal</h1>
			<input type="text" id="search" placeholder="Search for information...">
			<button id="search-btn">Search</button>
			<div id="results">
				<p>No results yet. Try searching for "quantum computing applications"</p>
			</div>
		</body>
		</html>
		""",
		"url": "https://research.example.com",
		"title": "Research Portal",
		"goal": "Find information about the latest breakthrough in quantum computing from 2024 and summarize the key findings",
		"screenshot": None,
		"tab_info": [{"id": 0, "url": "https://research.example.com", "title": "Research Portal", "is_active": True}],
	}
	
	logger.info(f"\nMock GAIA task goal: {mock_obs['goal']}")
	logger.info("Agent would perform actions like:")
	logger.info("1. Type 'quantum computing breakthrough 2024' in search box")
	logger.info("2. Click search button")
	logger.info("3. Navigate through results")
	logger.info("4. Extract and summarize information")


def run_gaia_subset():
	"""Run a specific subset of GAIA tasks for testing."""
	try:
		from agentlab.experiments.study import make_study
		from browser_use.agentlab import BrowserUseAgent, BrowserUseAgentArgs
		import json
		
		# Configure agent for GAIA
		agent_args = BrowserUseAgentArgs(
			model_name="gpt-4o",
			temperature=0.3,
			use_vision=True,
			use_thinking=True,
			enable_memory=True,
			planner_interval=2,
			max_actions_per_step=3
		)
		
		# Create study with specific task IDs (if available)
		# GAIA tasks are typically numbered like gaia.1, gaia.2, etc.
		study = make_study(
			benchmark="gaia",
			agent_args=[agent_args],
			comment="GAIA subset test",
			task_ids=["gaia.1", "gaia.2", "gaia.3"],  # First 3 tasks
		)
		
		logger.info("Running subset of GAIA tasks...")
		
		# Run with detailed logging
		results = []
		for task_id in study.task_ids:
			logger.info(f"\nRunning task: {task_id}")
			try:
				result = study.run_single_task(task_id)
				results.append(result)
				logger.info(f"Task {task_id} completed: {result.get('success', False)}")
			except Exception as e:
				logger.error(f"Error on task {task_id}: {e}")
				results.append({"task_id": task_id, "success": False, "error": str(e)})
		
		# Save results
		output_dir = Path("gaia_results")
		output_dir.mkdir(exist_ok=True)
		
		with open(output_dir / "browser_use_gaia_results.json", "w") as f:
			json.dump(results, f, indent=2)
		
		logger.info(f"\nResults saved to {output_dir / 'browser_use_gaia_results.json'}")
		
		return results
		
	except Exception as e:
		logger.error(f"Error in GAIA subset: {e}")
		return []


def analyze_gaia_results(results):
	"""Analyze and display GAIA benchmark results."""
	if not results:
		logger.info("No results to analyze")
		return
	
	logger.info("\n" + "="*50)
	logger.info("GAIA Benchmark Results Analysis")
	logger.info("="*50)
	
	# Calculate metrics
	total_tasks = len(results)
	successful_tasks = sum(1 for r in results if r.get('success', False))
	success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
	
	logger.info(f"Total tasks attempted: {total_tasks}")
	logger.info(f"Successful tasks: {successful_tasks}")
	logger.info(f"Success rate: {success_rate:.1f}%")
	
	# Average steps for successful tasks
	successful_steps = [r.get('steps', 0) for r in results if r.get('success', False)]
	if successful_steps:
		avg_steps = sum(successful_steps) / len(successful_steps)
		logger.info(f"Average steps for successful tasks: {avg_steps:.1f}")
	
	# Task categories (if available)
	categories = {}
	for result in results:
		category = result.get('category', 'unknown')
		if category not in categories:
			categories[category] = {'total': 0, 'success': 0}
		categories[category]['total'] += 1
		if result.get('success', False):
			categories[category]['success'] += 1
	
	if len(categories) > 1:
		logger.info("\nResults by category:")
		for cat, stats in categories.items():
			cat_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
			logger.info(f"  {cat}: {stats['success']}/{stats['total']} ({cat_rate:.1f}%)")


if __name__ == "__main__":
	# Check environment variables
	if not os.getenv("OPENAI_API_KEY"):
		logger.warning("OPENAI_API_KEY not set. Please set it to run GAIA benchmark.")
		logger.info("Options:")
		logger.info("1. Create a .env file with: OPENAI_API_KEY='your-api-key'")
		logger.info("2. Or export OPENAI_API_KEY='your-api-key'")
	
	# Run GAIA benchmark
	results = run_gaia_benchmark()
	
	# If no results from full benchmark, try subset
	if not results:
		logger.info("\nTrying to run a subset of GAIA tasks...")
		results = run_gaia_subset()
	
	# Analyze results
	if results:
		analyze_gaia_results(results)
	
	logger.info("\nGAIA benchmark example completed.")