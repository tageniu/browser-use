"""AgentLab configuration for Browser-Use agent."""

from browser_use.agentlab.agent_args import BrowserUseAgentArgs

# Default configuration
DEFAULT_BROWSER_USE_AGENT = BrowserUseAgentArgs(
	model_name="gpt-4o",
	temperature=0.7,
	use_vision=True,
	use_thinking=True,
	headless=True,
)

# Configuration for different benchmarks
WEBARENA_CONFIG = BrowserUseAgentArgs(
	model_name="gpt-4o",
	temperature=0.5,
	use_vision=True,
	use_thinking=True,
	use_accessibility_tree=True,
	max_actions_per_step=1,
	headless=True,
)

WORKARENA_CONFIG = BrowserUseAgentArgs(
	model_name="gpt-4o",
	temperature=0.5,
	use_vision=True,
	use_thinking=True,
	viewport_width=1280,
	viewport_height=960,
	headless=True,
)

MINIWOB_CONFIG = BrowserUseAgentArgs(
	model_name="gpt-4o-mini",
	temperature=0.3,
	use_vision=False,
	use_accessibility_tree=False,
	viewport_width=500,
	viewport_height=500,
	max_actions_per_step=1,
	headless=True,
)

# GAIA benchmark configuration
GAIA_CONFIG = BrowserUseAgentArgs(
	model_name="gpt-4o",
	temperature=0.3,  # Lower temperature for accuracy
	use_vision=True,
	use_thinking=True,
	use_accessibility_tree=True,
	enable_memory=True,  # Important for multi-step GAIA tasks
	max_actions_per_step=3,  # GAIA tasks may require multiple actions
	planner_interval=2,  # More frequent planning
	viewport_width=1280,
	viewport_height=960,
	headless=True,
)

# Reproducibility configuration
REPRODUCIBLE_CONFIG = BrowserUseAgentArgs(
	model_name="gpt-4o",
	temperature=0.0,  # Deterministic
	use_vision=True,
	use_thinking=True,
	headless=True,
)


def get_config_for_benchmark(benchmark: str) -> BrowserUseAgentArgs:
	"""Get the appropriate configuration for a given benchmark."""
	configs = {
		"webarena": WEBARENA_CONFIG,
		"workarena": WORKARENA_CONFIG,
		"miniwob": MINIWOB_CONFIG,
		"gaia": GAIA_CONFIG,
		"default": DEFAULT_BROWSER_USE_AGENT,
	}
	
	config = configs.get(benchmark.lower(), DEFAULT_BROWSER_USE_AGENT)
	# Create a copy to avoid modifying the original
	return config.model_copy()