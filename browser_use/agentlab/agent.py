"""Browser-Use Agent implementation for AgentLab."""

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel

from browser_use.agentlab.action_mapper import ActionMapper
from browser_use.agentlab.agent_args import BrowserUseAgentArgs
from browser_use.agentlab.observation_processor import ObservationProcessor
from browser_use.agent.service import Agent as BrowserUseAgentCore
from browser_use.agent.views import AgentOutput
from browser_use.browser import BrowserSession
from browser_use.controller.service import Controller
from browser_use.llm.openai.chat import ChatOpenAI

# Check if browsergym is available
try:
	from browsergym.core.agent import Agent
except ImportError:
	# Create a dummy Agent class for development
	class Agent:
		"""Dummy Agent for development when browsergym is not installed"""
		def __init__(self):
			pass
		
		def get_action(self, obs: dict) -> tuple[str, dict]:
			raise NotImplementedError

logger = logging.getLogger(__name__)


class BrowserUseAgent(Agent):
	"""Browser-Use agent implementation for AgentLab/BrowserGym."""
	
	def __init__(self, agent_args: BrowserUseAgentArgs):
		super().__init__()
		self.agent_args = agent_args
		self._browser_session = None
		self._agent_core = None
		self._action_mapper = ActionMapper(strategy=agent_args.action_mapping_strategy)
		self._observation_processor = ObservationProcessor(
			use_vision=agent_args.use_vision,
			use_accessibility_tree=agent_args.use_accessibility_tree
		)
		
		# Initialize LLM
		self._init_llm()
		
		# Event loop for async operations
		self._loop = None
		
	def _init_llm(self):
		"""Initialize the LLM client based on agent args."""
		# For now, using OpenAI as default
		# In full implementation, this would support multiple LLM providers
		self.llm = ChatOpenAI(
			model=self.agent_args.model_name,
			temperature=self.agent_args.temperature,
		)
		
	def _ensure_event_loop(self):
		"""Ensure we have an event loop for async operations."""
		try:
			self._loop = asyncio.get_running_loop()
		except RuntimeError:
			self._loop = asyncio.new_event_loop()
			asyncio.set_event_loop(self._loop)
			
	def _run_async(self, coro):
		"""Run an async coroutine in the event loop."""
		# Check if we're in an async context
		try:
			loop = asyncio.get_running_loop()
			# We're already in an event loop, schedule the coroutine
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as executor:
				future = executor.submit(asyncio.run, coro)
				return future.result()
		except RuntimeError:
			# No event loop running, we can use asyncio.run
			return asyncio.run(coro)
		
	def get_action(self, obs: dict) -> tuple[str, dict]:
		"""
		Get the next action given an observation from BrowserGym.
		
		Args:
			obs: Observation dictionary from BrowserGym containing:
				- dom_object: HTML content
				- screenshot: Screenshot data (if enabled)
				- url: Current URL
				- goal: Task goal
				- error: Any error messages
				- tab_info: Information about open tabs
				
		Returns:
			Tuple of (action_string, agent_info_dict)
		"""
		try:
			# Process observation into Browser-Use format
			processed_obs = self._observation_processor.process(obs)
			
			# Extract task if this is the first observation
			task = processed_obs.get('goal', '') or processed_obs.get('task', '')
			
			# Initialize Browser-Use agent if needed
			if self._agent_core is None:
				self._init_browser_use_agent(task, obs)
				
			# Get action from Browser-Use agent
			browser_use_action = self._get_browser_use_action(processed_obs)
			
			# Convert to BrowserGym action format
			browsergym_action = self._action_mapper.to_browsergym(
				browser_use_action,
				obs.get('dom_object', '')
			)
			
			# Prepare agent info
			agent_info = {
				'think': browser_use_action.get('thinking', ''),
				'action_type': browser_use_action.get('type', ''),
				'raw_action': browser_use_action,
				'browser_use_output': browser_use_action.get('output', {}),
			}
			
			logger.info(f"Action: {browsergym_action}")
			
			return browsergym_action, agent_info
			
		except Exception as e:
			logger.error(f"Error getting action: {e}", exc_info=True)
			# Return a no-op action on error
			return "noop()", {"error": str(e)}
			
	def _init_browser_use_agent(self, task: str, obs: dict):
		"""Initialize the Browser-Use agent core."""
		# If using BrowserGym's browser, we pass None and let Browser-Use know
		# to use the provided page/browser context
		browser = None if self.agent_args.use_browsergym_browser else BrowserSession()
		
		# Create controller
		controller = Controller()
		
		# Initialize Browser-Use agent
		self._agent_core = BrowserUseAgentCore(
			task=task,
			llm=self.llm,
			browser=browser,
			controller=controller,
			use_vision=self.agent_args.use_vision,
			use_thinking=self.agent_args.use_thinking,
			max_actions_per_step=self.agent_args.max_actions_per_step,
			enable_memory=self.agent_args.enable_memory,
			planner_interval=self.agent_args.planner_interval,
		)
		
	def _get_browser_use_action(self, processed_obs: dict) -> dict:
		"""Get action from Browser-Use agent."""
		# Create a mock browser state for Browser-Use
		# In full implementation, this would properly bridge the browser states
		browser_state = self._create_browser_state(processed_obs)
		
		# Run Browser-Use step (async operation)
		result = self._run_async(self._agent_core._step(browser_state))
		
		if isinstance(result, AgentOutput):
			# Extract action from AgentOutput
			action = self._extract_action_from_output(result)
			return action
		else:
			# Fallback for unexpected result type
			return {"type": "noop", "thinking": "Unexpected result type"}
			
	def _create_browser_state(self, obs: dict) -> dict:
		"""Create a browser state compatible with Browser-Use from BrowserGym observation."""
		# This is a simplified version - full implementation would properly
		# convert all BrowserGym observation fields to Browser-Use format
		return {
			'html': obs.get('dom_object', ''),
			'screenshot': obs.get('screenshot'),
			'url': obs.get('url', ''),
			'title': obs.get('title', ''),
			'tabs': obs.get('tab_info', []),
		}
		
	def _extract_action_from_output(self, output: AgentOutput) -> dict:
		"""Extract action information from Browser-Use AgentOutput."""
		# Check if output has actions attribute
		if hasattr(output, 'actions') and output.actions:
			last_action = output.actions[-1]
			# Handle different action formats
			if isinstance(last_action, dict):
				return {
					'type': last_action.get('type', 'unknown'),
					'params': last_action.get('params', {}),
					'thinking': output.current_state.thinking if hasattr(output, 'current_state') else '',
					'output': output.model_dump() if hasattr(output, 'model_dump') else {},
				}
			else:
				# Handle ActionModel or other formats
				return {
					'type': getattr(last_action, 'type', 'unknown'),
					'params': getattr(last_action, 'params', {}),
					'thinking': output.current_state.thinking if hasattr(output, 'current_state') else '',
					'output': output.model_dump() if hasattr(output, 'model_dump') else {},
				}
		else:
			return {
				'type': 'noop',
				'thinking': getattr(output.current_state, 'thinking', '') if hasattr(output, 'current_state') else '',
				'output': output.model_dump() if hasattr(output, 'model_dump') else {},
			}
			
	def reset(self):
		"""Reset the agent state."""
		self._agent_core = None
		self._browser_session = None
		if self._loop and not self._loop.is_closed():
			self._loop.close()
		self._loop = None