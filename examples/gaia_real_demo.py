"""Real GAIA demo with actual web navigation using Browser-Use."""

import asyncio
import json
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from browser_use import Agent
from browser_use.browser import Browser, BrowserConfig
from browser_use.llm.openai.chat import ChatOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_real_gaia_task():
	"""Run a real GAIA-style web navigation task."""
	
	# Create a simple browser config to avoid compatibility issues
	browser_config = BrowserConfig(
		headless=False,  # Show browser for demo
		viewport={'width': 1280, 'height': 720},
	)
	
	browser = Browser(config=browser_config)
	
	# Initialize LLM
	llm = ChatOpenAI(
		model="gpt-4o",
		temperature=0.3,  # Lower temperature for accuracy
		api_key=os.getenv("OPENAI_API_KEY")
	)
	
	# Real GAIA-style task: Search for current information
	task = """
	Go to Wikipedia and find information about the 2024 Nobel Prize in Physics winners.
	Extract their names and the reason they won the prize.
	Return a summary with:
	1. The winners' names
	2. Their affiliations
	3. The reason for the award
	"""
	
	logger.info("="*60)
	logger.info("Running Real GAIA Task: Nobel Prize Information Retrieval")
	logger.info("="*60)
	logger.info(f"Task: {task.strip()}")
	logger.info("")
	
	# Create agent
	agent = Agent(
		task=task,
		llm=llm,
		browser=browser,
		use_vision=True,
		use_thinking=True,
		max_actions_per_step=3,
		max_failures=5,
		save_conversation_path="gaia_nobel_prize_search.md",
	)
	
	# Run the agent
	start_time = datetime.now()
	logger.info(f"Starting at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
	
	try:
		history = await agent.run(max_steps=20)
		
		# Calculate execution time
		end_time = datetime.now()
		execution_time = (end_time - start_time).total_seconds()
		
		logger.info(f"\nTask completed in {execution_time:.2f} seconds")
		logger.info(f"Total steps taken: {len(history) if history else 0}")
		
		# Extract the result
		if history and len(history) > 0:
			# Look for the final result
			final_result = None
			for entry in reversed(history):
				if hasattr(entry, 'result') and entry.result:
					if hasattr(entry.result, 'extracted_content'):
						final_result = entry.result.extracted_content
						break
					elif hasattr(entry.result, 'text'):
						final_result = entry.result.text
						break
			
			if final_result:
				logger.info("\n" + "="*60)
				logger.info("EXTRACTED INFORMATION:")
				logger.info("="*60)
				logger.info(final_result)
			else:
				logger.info("\nNo specific result extracted, check the conversation log.")
		
		# Save results
		results = {
			"task": "Nobel Prize 2024 Physics Winners",
			"execution_time_seconds": execution_time,
			"steps_taken": len(history) if history else 0,
			"timestamp": datetime.now().isoformat(),
			"success": True,
			"result": final_result if 'final_result' in locals() else "See conversation log"
		}
		
		with open("gaia_real_results.json", "w") as f:
			json.dump(results, f, indent=2)
		
		logger.info(f"\nResults saved to gaia_real_results.json")
		logger.info(f"Conversation log saved to gaia_nobel_prize_search.md")
		
	except Exception as e:
		logger.error(f"Error during task execution: {e}")
		results = {
			"task": "Nobel Prize 2024 Physics Winners",
			"error": str(e),
			"timestamp": datetime.now().isoformat(),
			"success": False
		}
		with open("gaia_real_results.json", "w") as f:
			json.dump(results, f, indent=2)
	
	finally:
		# Close browser
		await browser.close()


async def run_multiple_gaia_tasks():
	"""Run multiple GAIA-style tasks to demonstrate capabilities."""
	
	tasks = [
		{
			"id": "current_weather",
			"task": "Go to weather.com and find the current temperature in New York City. Return just the temperature in Fahrenheit.",
			"max_steps": 10
		},
		{
			"id": "python_version", 
			"task": "Go to python.org and find the latest stable Python version number. Return just the version number (e.g., 3.12.0).",
			"max_steps": 10
		},
		{
			"id": "stock_price",
			"task": "Go to finance.yahoo.com and find the current stock price of Apple (AAPL). Return the current price.",
			"max_steps": 15
		}
	]
	
	browser_config = BrowserConfig(
		headless=True,  # Run headless for multiple tasks
		viewport_width=1280,
		viewport_height=960,
	)
	
	browser = Browser(config=browser_config)
	llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
	
	all_results = []
	
	for task_info in tasks:
		logger.info(f"\n{'='*60}")
		logger.info(f"Running Task: {task_info['id']}")
		logger.info(f"{'='*60}")
		
		agent = Agent(
			task=task_info['task'],
			llm=llm,
			browser=browser,
			use_vision=True,
			max_actions_per_step=2,
		)
		
		start_time = datetime.now()
		
		try:
			history = await agent.run(max_steps=task_info['max_steps'])
			execution_time = (datetime.now() - start_time).total_seconds()
			
			result = {
				"task_id": task_info['id'],
				"success": True,
				"execution_time": execution_time,
				"steps": len(history) if history else 0
			}
			
			logger.info(f"✓ Completed in {execution_time:.2f}s with {result['steps']} steps")
			
		except Exception as e:
			logger.error(f"✗ Failed: {e}")
			result = {
				"task_id": task_info['id'],
				"success": False,
				"error": str(e)
			}
		
		all_results.append(result)
	
	await browser.close()
	
	# Summary
	logger.info(f"\n{'='*60}")
	logger.info("SUMMARY")
	logger.info(f"{'='*60}")
	successful = sum(1 for r in all_results if r.get('success', False))
	logger.info(f"Tasks completed: {successful}/{len(tasks)}")
	
	with open("gaia_multiple_tasks_results.json", "w") as f:
		json.dump(all_results, f, indent=2)


def main():
	logger.info("Browser-Use GAIA Demo")
	asyncio.run(run_real_gaia_task())
	
	# Uncomment to run multiple tasks
	# asyncio.run(run_multiple_gaia_tasks())


if __name__ == "__main__":
	main()