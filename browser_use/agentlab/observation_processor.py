"""Processes BrowserGym observations for Browser-Use."""

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ObservationProcessor:
	"""Converts BrowserGym observations to Browser-Use format."""
	
	def __init__(self, use_vision: bool = True, use_accessibility_tree: bool = True):
		"""
		Initialize the observation processor.
		
		Args:
			use_vision: Whether to include screenshots
			use_accessibility_tree: Whether to process accessibility info
		"""
		self.use_vision = use_vision
		self.use_accessibility_tree = use_accessibility_tree
		
	def process(self, obs: dict) -> dict:
		"""
		Process BrowserGym observation into Browser-Use format.
		
		Args:
			obs: BrowserGym observation dictionary
			
		Returns:
			Processed observation for Browser-Use
		"""
		processed = {}
		
		# Basic fields
		processed['url'] = obs.get('url', '')
		processed['title'] = obs.get('title', '')
		processed['goal'] = obs.get('goal', '')
		processed['task'] = obs.get('goal', '')  # Browser-Use uses 'task'
		
		# HTML content
		dom_object = obs.get('dom_object', '')
		if isinstance(dom_object, str):
			processed['html'] = dom_object
		else:
			# Handle structured DOM if provided
			processed['html'] = self._extract_html_from_dom(dom_object)
			
		# Screenshot
		if self.use_vision and 'screenshot' in obs:
			processed['screenshot'] = self._process_screenshot(obs['screenshot'])
			
		# Tab information
		tab_info = obs.get('tab_info', [])
		processed['tabs'] = self._process_tabs(tab_info)
		
		# Error information
		if 'error' in obs:
			processed['error'] = obs['error']
			
		# Accessibility tree
		if self.use_accessibility_tree and 'axtree_object' in obs:
			processed['accessibility_tree'] = obs['axtree_object']
			
		# Additional metadata
		processed['metadata'] = {
			'focused_element': obs.get('focused_element_bid'),
			'viewport': obs.get('viewport_size', {}),
			'scroll_position': obs.get('scroll_position', {}),
		}
		
		return processed
		
	def _extract_html_from_dom(self, dom_object: Any) -> str:
		"""Extract HTML string from structured DOM object."""
		if isinstance(dom_object, str):
			return dom_object
		elif hasattr(dom_object, 'html'):
			return dom_object.html
		elif isinstance(dom_object, dict) and 'html' in dom_object:
			return dom_object['html']
		else:
			# Try to convert to string
			return str(dom_object)
			
	def _process_screenshot(self, screenshot_data: Any) -> str | None:
		"""Process screenshot data into base64 string."""
		if screenshot_data is None:
			return None
			
		if isinstance(screenshot_data, str):
			# Already base64 encoded
			return screenshot_data
		elif isinstance(screenshot_data, bytes):
			# Convert bytes to base64
			return base64.b64encode(screenshot_data).decode('utf-8')
		elif hasattr(screenshot_data, 'read'):
			# File-like object
			data = screenshot_data.read()
			if isinstance(data, bytes):
				return base64.b64encode(data).decode('utf-8')
			return data
		else:
			logger.warning(f"Unknown screenshot data type: {type(screenshot_data)}")
			return None
			
	def _process_tabs(self, tab_info: list) -> list[dict]:
		"""Process tab information."""
		tabs = []
		
		for tab in tab_info:
			if isinstance(tab, dict):
				tabs.append({
					'id': tab.get('id', len(tabs)),
					'url': tab.get('url', ''),
					'title': tab.get('title', ''),
					'active': tab.get('is_active', False),
				})
			else:
				# Handle other tab formats
				tabs.append({
					'id': len(tabs),
					'info': str(tab),
				})
				
		return tabs
		
	def create_browser_state(self, obs: dict) -> dict:
		"""
		Create a browser state summary compatible with Browser-Use.
		
		Args:
			obs: BrowserGym observation
			
		Returns:
			Browser state dictionary
		"""
		processed = self.process(obs)
		
		return {
			'url': processed['url'],
			'title': processed['title'],
			'html': processed['html'],
			'screenshot': processed.get('screenshot'),
			'tabs': processed['tabs'],
			'error': processed.get('error'),
			'metadata': processed.get('metadata', {}),
		}