"""AgentLab agent arguments for Browser-Use integration."""

from pydantic import ConfigDict, Field

# Check if agentlab is available
try:
	from agentlab.agents.agent_args import AgentArgs
except ImportError:
	# Create a dummy AgentArgs class for development
	from pydantic import BaseModel
	
	class AgentArgs(BaseModel):
		"""Dummy AgentArgs for development when agentlab is not installed"""
		pass


class BrowserUseAgentArgs(AgentArgs):
	"""Configuration for Browser-Use agent in AgentLab."""
	
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	# Browser-Use specific settings
	headless: bool = Field(True, description="Run browser in headless mode")
	browser_type: str = Field("chromium", description="Browser type: chromium, firefox, or webkit")
	viewport_width: int = Field(1280, description="Browser viewport width")
	viewport_height: int = Field(720, description="Browser viewport height")
	
	# LLM settings
	model_name: str = Field("gpt-4o", description="LLM model to use")
	temperature: float = Field(0.7, description="Temperature for LLM responses")
	max_retries: int = Field(3, description="Maximum retries for failed actions")
	
	# Agent behavior
	use_vision: bool = Field(True, description="Use screenshot analysis")
	use_accessibility_tree: bool = Field(True, description="Include accessibility tree in observations")
	use_thinking: bool = Field(True, description="Enable agent thinking process")
	max_actions_per_step: int = Field(1, description="Maximum actions per step")
	
	# Integration settings
	use_browsergym_browser: bool = Field(
		True, 
		description="Use BrowserGym's browser instance instead of creating our own"
	)
	action_mapping_strategy: str = Field(
		"element_id", 
		description="Strategy for mapping actions: 'element_id' or 'selector'"
	)
	
	# Memory and planning
	enable_memory: bool = Field(True, description="Enable agent memory")
	planner_interval: int = Field(5, description="Run planner every N steps")
	
	def set_reproducibility_mode(self):
		"""Set parameters for reproducible runs."""
		self.temperature = 0.0
		
	def set_benchmark(self, benchmark: str, demo_mode: bool = False):
		"""Adjust settings for specific benchmarks."""
		if benchmark == "workarena":
			self.use_vision = True
			self.viewport_width = 1280
			self.viewport_height = 960
		elif benchmark == "miniwob":
			self.viewport_width = 500
			self.viewport_height = 500
			self.use_accessibility_tree = False
		elif benchmark == "webarena":
			self.use_vision = True
			self.max_actions_per_step = 1
			
	def make_agent(self):
		"""Create a Browser-Use agent instance."""
		from browser_use.agentlab.agent import BrowserUseAgent
		return BrowserUseAgent(self)