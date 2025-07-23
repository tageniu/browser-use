"""Maps between Browser-Use and BrowserGym action formats."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ActionMapper:
	"""Converts actions between Browser-Use and BrowserGym formats."""
	
	def __init__(self, strategy: str = "element_id"):
		"""
		Initialize the action mapper.
		
		Args:
			strategy: Mapping strategy - "element_id" or "selector"
		"""
		self.strategy = strategy
		
	def to_browsergym(self, browser_use_action: dict, dom_html: str = "") -> str:
		"""
		Convert Browser-Use action to BrowserGym action string.
		
		Args:
			browser_use_action: Action dict from Browser-Use
			dom_html: HTML content for element mapping
			
		Returns:
			BrowserGym action string
		"""
		action_type = browser_use_action.get('type', '')
		params = browser_use_action.get('params', {})
		
		try:
			if action_type == 'click':
				return self._convert_click(params, dom_html)
			elif action_type == 'type' or action_type == 'input_text':
				return self._convert_type(params, dom_html)
			elif action_type == 'scroll':
				return self._convert_scroll(params)
			elif action_type == 'go_to_url':
				return self._convert_go_to_url(params)
			elif action_type == 'done':
				return self._convert_done(params)
			elif action_type == 'press':
				return self._convert_press(params, dom_html)
			elif action_type == 'drag_drop':
				return self._convert_drag_drop(params, dom_html)
			elif action_type == 'hover':
				return self._convert_hover(params, dom_html)
			elif action_type == 'select':
				return self._convert_select(params, dom_html)
			elif action_type == 'switch_tab':
				return self._convert_switch_tab(params)
			elif action_type == 'open_tab':
				return self._convert_open_tab(params)
			elif action_type == 'close_tab':
				return self._convert_close_tab(params)
			else:
				logger.warning(f"Unknown action type: {action_type}")
				return "noop()"
				
		except Exception as e:
			logger.error(f"Error converting action: {e}", exc_info=True)
			return "noop()"
			
	def _convert_click(self, params: dict, dom_html: str) -> str:
		"""Convert click action."""
		if self.strategy == "element_id":
			# Try to find element ID (bid) from selector
			selector = params.get('selector', '')
			index = params.get('index')
			
			if index is not None:
				return f'click({index})'
			elif selector:
				# In real implementation, parse DOM to find bid
				# For now, pass the selector as-is
				return f'click("{selector}")'
			else:
				# Try coordinates
				x = params.get('x')
				y = params.get('y')
				if x is not None and y is not None:
					return f'mouse_click({x}, {y})'
					
		return "noop()"
		
	def _convert_type(self, params: dict, dom_html: str) -> str:
		"""Convert type/fill action."""
		text = params.get('text', '')
		selector = params.get('selector', '')
		index = params.get('index')
		
		if index is not None:
			return f'fill({index}, "{text}")'
		elif selector:
			return f'fill("{selector}", "{text}")'
			
		return "noop()"
		
	def _convert_scroll(self, params: dict) -> str:
		"""Convert scroll action."""
		delta_x = params.get('delta_x', 0)
		delta_y = params.get('delta_y', params.get('amount', 0))
		
		return f'scroll({delta_x}, {delta_y})'
		
	def _convert_go_to_url(self, params: dict) -> str:
		"""Convert navigation action."""
		url = params.get('url', '')
		return f'goto("{url}")'
		
	def _convert_done(self, params: dict) -> str:
		"""Convert done action."""
		text = params.get('text', 'Task completed')
		return f'report_answer("{text}")'
		
	def _convert_press(self, params: dict, dom_html: str) -> str:
		"""Convert key press action."""
		keys = params.get('keys', '')
		selector = params.get('selector', '')
		index = params.get('index')
		
		if index is not None:
			return f'press({index}, "{keys}")'
		elif selector:
			return f'press("{selector}", "{keys}")'
		else:
			# Global key press
			return f'keyboard_type("{keys}")'
			
	def _convert_drag_drop(self, params: dict, dom_html: str) -> str:
		"""Convert drag and drop action."""
		# Try element-based first
		source = params.get('source_selector') or params.get('element_source')
		target = params.get('target_selector') or params.get('element_target')
		
		if source and target:
			return f'drag_and_drop("{source}", "{target}")'
			
		# Fall back to coordinates
		source_x = params.get('source_x') or params.get('coord_source_x')
		source_y = params.get('source_y') or params.get('coord_source_y')
		target_x = params.get('target_x') or params.get('coord_target_x')
		target_y = params.get('target_y') or params.get('coord_target_y')
		
		if all(coord is not None for coord in [source_x, source_y, target_x, target_y]):
			return f'mouse_drag_and_drop({source_x}, {source_y}, {target_x}, {target_y})'
			
		return "noop()"
		
	def _convert_hover(self, params: dict, dom_html: str) -> str:
		"""Convert hover action."""
		selector = params.get('selector', '')
		index = params.get('index')
		
		if index is not None:
			return f'hover({index})'
		elif selector:
			return f'hover("{selector}")'
			
		return "noop()"
		
	def _convert_select(self, params: dict, dom_html: str) -> str:
		"""Convert select option action."""
		selector = params.get('selector', '')
		index = params.get('index')
		options = params.get('options', [])
		
		if isinstance(options, list):
			options_str = json.dumps(options)
		else:
			options_str = json.dumps([str(options)])
			
		if index is not None:
			return f'select_option({index}, {options_str})'
		elif selector:
			return f'select_option("{selector}", {options_str})'
			
		return "noop()"
		
	def _convert_switch_tab(self, params: dict) -> str:
		"""Convert switch tab action."""
		page_id = params.get('page_id') or params.get('tab_id') or 0
		return f'switch_tab({page_id})'
		
	def _convert_open_tab(self, params: dict) -> str:
		"""Convert open tab action."""
		url = params.get('url', '')
		return f'new_tab("{url}")'
		
	def _convert_close_tab(self, params: dict) -> str:
		"""Convert close tab action."""
		page_id = params.get('page_id') or params.get('tab_id') or 0
		return f'close_tab({page_id})'
		
	def from_browsergym(self, browsergym_action: str) -> dict:
		"""
		Convert BrowserGym action string to Browser-Use action dict.
		
		Args:
			browsergym_action: BrowserGym action string
			
		Returns:
			Browser-Use action dictionary
		"""
		# Parse the action string
		match = re.match(r'(\w+)\((.*)\)', browsergym_action)
		if not match:
			return {"type": "noop", "params": {}}
			
		action_name = match.group(1)
		args_str = match.group(2)
		
		# Parse arguments
		try:
			# Simple argument parsing - in real implementation would be more robust
			args = self._parse_args(args_str)
		except Exception as e:
			logger.error(f"Error parsing arguments: {e}")
			args = []
			
		# Convert to Browser-Use format
		if action_name == 'click':
			return self._parse_click(args)
		elif action_name == 'fill':
			return self._parse_fill(args)
		elif action_name == 'scroll':
			return self._parse_scroll(args)
		elif action_name == 'goto':
			return self._parse_goto(args)
		elif action_name == 'keyboard_type':
			return self._parse_keyboard_type(args)
		else:
			return {"type": action_name, "params": {"args": args}}
			
	def _parse_args(self, args_str: str) -> list:
		"""Parse action arguments from string."""
		if not args_str.strip():
			return []
			
		# Simple CSV parsing - would need proper parsing for complex cases
		args = []
		current = ''
		in_quotes = False
		quote_char = None
		
		for char in args_str:
			if char in '"\'':
				if not in_quotes:
					in_quotes = True
					quote_char = char
				elif char == quote_char:
					in_quotes = False
					quote_char = None
				else:
					current += char
			elif char == ',' and not in_quotes:
				args.append(current.strip())
				current = ''
			else:
				current += char
				
		if current:
			args.append(current.strip())
			
		# Convert numeric strings to numbers
		parsed_args = []
		for arg in args:
			try:
				parsed_args.append(int(arg))
			except ValueError:
				try:
					parsed_args.append(float(arg))
				except ValueError:
					# Remove quotes if present
					if arg.startswith('"') and arg.endswith('"'):
						arg = arg[1:-1]
					elif arg.startswith("'") and arg.endswith("'"):
						arg = arg[1:-1]
					parsed_args.append(arg)
					
		return parsed_args
		
	def _parse_click(self, args: list) -> dict:
		"""Parse click action."""
		if not args:
			return {"type": "click", "params": {}}
			
		if isinstance(args[0], int):
			return {"type": "click", "params": {"index": args[0]}}
		else:
			return {"type": "click", "params": {"selector": args[0]}}
			
	def _parse_fill(self, args: list) -> dict:
		"""Parse fill action."""
		if len(args) < 2:
			return {"type": "type", "params": {}}
			
		if isinstance(args[0], int):
			return {"type": "type", "params": {"index": args[0], "text": str(args[1])}}
		else:
			return {"type": "type", "params": {"selector": args[0], "text": str(args[1])}}
			
	def _parse_scroll(self, args: list) -> dict:
		"""Parse scroll action."""
		if len(args) >= 2:
			return {"type": "scroll", "params": {"delta_x": args[0], "delta_y": args[1]}}
		elif len(args) == 1:
			return {"type": "scroll", "params": {"delta_y": args[0]}}
		else:
			return {"type": "scroll", "params": {"delta_y": 300}}
			
	def _parse_goto(self, args: list) -> dict:
		"""Parse goto action."""
		if args:
			return {"type": "go_to_url", "params": {"url": args[0]}}
		else:
			return {"type": "go_to_url", "params": {"url": ""}}
			
	def _parse_keyboard_type(self, args: list) -> dict:
		"""Parse keyboard type action."""
		if args:
			return {"type": "press", "params": {"keys": args[0]}}
		else:
			return {"type": "press", "params": {"keys": ""}}