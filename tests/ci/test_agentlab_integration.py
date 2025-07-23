"""Tests for AgentLab integration."""

import pytest
from unittest.mock import Mock, patch

from browser_use.agentlab.action_mapper import ActionMapper
from browser_use.agentlab.agent_args import BrowserUseAgentArgs
from browser_use.agentlab.observation_processor import ObservationProcessor


class TestActionMapper:
	"""Test action mapping between Browser-Use and BrowserGym."""
	
	def setup_method(self):
		"""Set up test fixtures."""
		self.mapper = ActionMapper(strategy="element_id")
		
	def test_click_with_index(self):
		"""Test converting click action with index."""
		browser_use_action = {
			"type": "click",
			"params": {"index": 5}
		}
		result = self.mapper.to_browsergym(browser_use_action)
		assert result == "click(5)"
		
	def test_click_with_selector(self):
		"""Test converting click action with selector."""
		browser_use_action = {
			"type": "click",
			"params": {"selector": "#submit-button"}
		}
		result = self.mapper.to_browsergym(browser_use_action)
		assert result == 'click("#submit-button")'
		
	def test_click_with_coordinates(self):
		"""Test converting click action with coordinates."""
		browser_use_action = {
			"type": "click",
			"params": {"x": 100, "y": 200}
		}
		result = self.mapper.to_browsergym(browser_use_action)
		assert result == "mouse_click(100, 200)"
		
	def test_type_action(self):
		"""Test converting type action."""
		browser_use_action = {
			"type": "type",
			"params": {"index": 3, "text": "Hello World"}
		}
		result = self.mapper.to_browsergym(browser_use_action)
		assert result == 'fill(3, "Hello World")'
		
	def test_scroll_action(self):
		"""Test converting scroll action."""
		browser_use_action = {
			"type": "scroll",
			"params": {"delta_y": 300}
		}
		result = self.mapper.to_browsergym(browser_use_action)
		assert result == "scroll(0, 300)"
		
	def test_go_to_url_action(self):
		"""Test converting navigation action."""
		browser_use_action = {
			"type": "go_to_url",
			"params": {"url": "https://example.com"}
		}
		result = self.mapper.to_browsergym(browser_use_action)
		assert result == 'goto("https://example.com")'
		
	def test_unknown_action(self):
		"""Test handling unknown action type."""
		browser_use_action = {
			"type": "unknown_action",
			"params": {}
		}
		result = self.mapper.to_browsergym(browser_use_action)
		assert result == "noop()"
		
	def test_from_browsergym_click(self):
		"""Test parsing BrowserGym click action."""
		result = self.mapper.from_browsergym('click(5)')
		assert result == {"type": "click", "params": {"index": 5}}
		
	def test_from_browsergym_fill(self):
		"""Test parsing BrowserGym fill action."""
		result = self.mapper.from_browsergym('fill(3, "test text")')
		assert result == {"type": "type", "params": {"index": 3, "text": "test text"}}


class TestObservationProcessor:
	"""Test observation processing."""
	
	def setup_method(self):
		"""Set up test fixtures."""
		self.processor = ObservationProcessor(use_vision=True, use_accessibility_tree=True)
		
	def test_basic_observation_processing(self):
		"""Test processing basic observation fields."""
		obs = {
			"url": "https://example.com",
			"title": "Example Page",
			"goal": "Click the button",
			"dom_object": "<html><body>Hello</body></html>",
		}
		
		result = self.processor.process(obs)
		
		assert result["url"] == "https://example.com"
		assert result["title"] == "Example Page"
		assert result["goal"] == "Click the button"
		assert result["task"] == "Click the button"
		assert result["html"] == "<html><body>Hello</body></html>"
		
	def test_screenshot_processing_base64(self):
		"""Test processing base64 screenshot."""
		obs = {
			"screenshot": "base64encodeddata"
		}
		
		result = self.processor.process(obs)
		assert result["screenshot"] == "base64encodeddata"
		
	def test_screenshot_processing_bytes(self):
		"""Test processing bytes screenshot."""
		obs = {
			"screenshot": b"imagedata"
		}
		
		result = self.processor.process(obs)
		# Should be base64 encoded
		assert isinstance(result["screenshot"], str)
		assert len(result["screenshot"]) > 0
		
	def test_tab_processing(self):
		"""Test processing tab information."""
		obs = {
			"tab_info": [
				{"id": 0, "url": "https://example.com", "title": "Tab 1", "is_active": True},
				{"id": 1, "url": "https://example.org", "title": "Tab 2", "is_active": False},
			]
		}
		
		result = self.processor.process(obs)
		
		assert len(result["tabs"]) == 2
		assert result["tabs"][0]["id"] == 0
		assert result["tabs"][0]["active"] == True
		assert result["tabs"][1]["active"] == False
		
	def test_error_handling(self):
		"""Test error information is preserved."""
		obs = {
			"error": "Something went wrong"
		}
		
		result = self.processor.process(obs)
		assert result["error"] == "Something went wrong"


class TestBrowserUseAgentArgs:
	"""Test agent arguments configuration."""
	
	def test_default_values(self):
		"""Test default argument values."""
		args = BrowserUseAgentArgs()
		
		assert args.headless == True
		assert args.browser_type == "chromium"
		assert args.viewport_width == 1280
		assert args.viewport_height == 720
		assert args.use_vision == True
		assert args.use_thinking == True
		
	def test_reproducibility_mode(self):
		"""Test reproducibility mode sets temperature to 0."""
		args = BrowserUseAgentArgs(temperature=0.7)
		assert args.temperature == 0.7
		
		args.set_reproducibility_mode()
		assert args.temperature == 0.0
		
	def test_benchmark_settings(self):
		"""Test benchmark-specific settings."""
		args = BrowserUseAgentArgs()
		
		# Test WebArena settings
		args.set_benchmark("webarena")
		assert args.use_vision == True
		assert args.max_actions_per_step == 1
		
		# Test MiniWoB settings
		args.set_benchmark("miniwob")
		assert args.viewport_width == 500
		assert args.viewport_height == 500
		assert args.use_accessibility_tree == False
		
		# Test WorkArena settings
		args.set_benchmark("workarena")
		assert args.use_vision == True
		assert args.viewport_width == 1280
		assert args.viewport_height == 960


@pytest.mark.asyncio
async def test_agent_creation():
	"""Test creating a Browser-Use agent for AgentLab."""
	from browser_use.agentlab import BrowserUseAgent
	
	args = BrowserUseAgentArgs(
		model_name="gpt-4o",
		temperature=0.5,
		use_vision=True,
	)
	
	# Mock the LLM client to avoid API calls
	with patch('browser_use.agentlab.agent.ChatOpenAI'):
		agent = BrowserUseAgent(args)
		
		assert agent.agent_args == args
		assert agent._action_mapper.strategy == "element_id"
		assert agent._observation_processor.use_vision == True


@pytest.mark.asyncio  
async def test_agent_get_action():
	"""Test agent get_action method."""
	from browser_use.agentlab import BrowserUseAgent
	
	args = BrowserUseAgentArgs(
		model_name="gpt-4o",
		use_browsergym_browser=True,
	)
	
	# Mock dependencies
	with patch('browser_use.agentlab.agent.ChatOpenAI'):
		agent = BrowserUseAgent(args)
		
		# Mock the Browser-Use agent core
		mock_agent_core = Mock()
		mock_output = Mock()
		mock_output.actions = [{"type": "click", "params": {"index": 5}}]
		mock_output.current_state.thinking = "I should click the button"
		mock_output.model_dump.return_value = {}
		
		# Create an async mock for _step
		async def mock_step(browser_state):
			return mock_output
		
		mock_agent_core._step = mock_step
		agent._agent_core = mock_agent_core
		
		# Create test observation
		obs = {
			"dom_object": "<html><body><button>Click me</button></body></html>",
			"url": "https://example.com",
			"goal": "Click the button",
		}
		
		# Get action
		action, info = agent.get_action(obs)
		
		# Check what we got
		print(f"Action: {action}")
		print(f"Info: {info}")
		
		# Should convert to BrowserGym format
		assert action == "click(5)"
		assert info["think"] == "I should click the button"
		assert info["action_type"] == "click"