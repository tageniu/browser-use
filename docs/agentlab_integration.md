# AgentLab Integration

Browser-Use now supports integration with [AgentLab](https://github.com/ServiceNow/AgentLab/), ServiceNow's framework for developing and benchmarking web agents.

## Overview

The integration allows you to:
- Run Browser-Use as an agent in AgentLab benchmarks
- Evaluate Browser-Use on WebArena, WorkArena, MiniWoB, and other benchmarks
- Compare performance against other agents on the AgentLab leaderboard
- Leverage AgentLab's parallel execution and reproducibility features

## Installation

First, install AgentLab:

```bash
pip install agentlab
```

The Browser-Use AgentLab integration is included in the main package.

## Quick Start

### Basic Usage

```python
from agentlab.experiments.study import make_study
from browser_use.agentlab import BrowserUseAgent, BrowserUseAgentArgs

# Configure the agent
agent_args = BrowserUseAgentArgs(
    model_name="gpt-4o",
    temperature=0.7,
    use_vision=True,
    use_thinking=True,
)

# Create and run a study
study = make_study(
    benchmark="webarena",
    agent_args=[agent_args],
    comment="Testing Browser-Use on WebArena"
)

# Run with parallel execution
study.run(n_jobs=5)  # Run 5 tasks in parallel

# Note: If n_jobs parameter is not supported in study.run(), 
# you may need to pass it to make_study() or use study.set_n_jobs()
```

### Using Benchmark-Specific Configurations

```python
from browser_use.agentlab.config import get_config_for_benchmark

# Get optimized configuration for a benchmark
agent_args = get_config_for_benchmark("workarena")

# Or customize it
agent_args = get_config_for_benchmark("webarena")
agent_args.temperature = 0.3
agent_args.max_actions_per_step = 2
```

### GAIA Benchmark

The GAIA (General AI Assistant) benchmark tests agents on complex, real-world tasks:

```python
from browser_use.agentlab.config import get_config_for_benchmark

# GAIA tasks often require multiple steps and memory
agent_args = get_config_for_benchmark("gaia")

# GAIA-optimized settings include:
# - enable_memory=True for multi-step tasks
# - max_actions_per_step=3 for complex interactions
# - planner_interval=2 for frequent re-planning
# - temperature=0.3 for accuracy

study = make_study(
    benchmark="gaia",
    agent_args=[agent_args],
    comment="Browser-Use on GAIA benchmark"
)
```

Example GAIA tasks include:
- Web research and information synthesis
- Data download and analysis
- Multi-step problem solving
- Cross-website information gathering

### Running a Specific Task

```python
from browser_use.agentlab import BrowserUseAgent
from agentlab.experiments import study_generators

# Create agent
agent = BrowserUseAgent(agent_args)

# Get a specific task environment
env = study_generators.get_benchmark_env("webarena", task_id="webarena.task_123")

# Run the agent
obs, info = env.reset()
done = False

while not done:
    action, agent_info = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
```

## Configuration Options

### BrowserUseAgentArgs

The `BrowserUseAgentArgs` class extends AgentLab's `AgentArgs` with Browser-Use specific settings:

```python
class BrowserUseAgentArgs(AgentArgs):
    # Browser settings
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    viewport_width: int = 1280
    viewport_height: int = 720
    
    # LLM settings
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_retries: int = 3
    
    # Agent behavior
    use_vision: bool = True
    use_accessibility_tree: bool = True
    use_thinking: bool = True
    max_actions_per_step: int = 1
    
    # Memory and planning
    enable_memory: bool = True
    planner_interval: int = 5
    
    # Integration settings
    use_browsergym_browser: bool = True
    action_mapping_strategy: str = "element_id"
```

### Pre-configured Settings

The integration includes optimized configurations for different benchmarks:

- **WEBARENA_CONFIG**: Optimized for WebArena tasks
- **WORKARENA_CONFIG**: Optimized for WorkArena enterprise tasks
- **MINIWOB_CONFIG**: Optimized for MiniWoB micro-tasks
- **REPRODUCIBLE_CONFIG**: Deterministic settings for reproducible runs

## Architecture

### Action Mapping

The integration handles conversion between Browser-Use and BrowserGym action formats:

**Browser-Use Action:**
```python
{
    "type": "click",
    "params": {
        "selector": "#submit-button"
    }
}
```

**BrowserGym Action:**
```
click("#submit-button")
```

### Observation Processing

BrowserGym observations are processed into Browser-Use format:

**BrowserGym Observation:**
```python
{
    "dom_object": "<html>...",
    "screenshot": base64_data,
    "url": "https://example.com",
    "goal": "Click the submit button"
}
```

**Browser-Use Format:**
```python
{
    "html": "<html>...",
    "screenshot": base64_data,
    "url": "https://example.com",
    "task": "Click the submit button",
    "tabs": [...]
}
```

## Advanced Usage

### Custom Action Mapping Strategy

```python
from browser_use.agentlab.action_mapper import ActionMapper

class CustomActionMapper(ActionMapper):
    def to_browsergym(self, browser_use_action, dom_html):
        # Custom conversion logic
        pass

# Use custom mapper
agent._action_mapper = CustomActionMapper()
```

### Extending the Agent

```python
from browser_use.agentlab import BrowserUseAgent

class CustomBrowserUseAgent(BrowserUseAgent):
    def get_action(self, obs):
        # Pre-process observation
        obs = self.preprocess(obs)
        
        # Get action from parent
        action, info = super().get_action(obs)
        
        # Post-process action
        action = self.postprocess(action)
        
        return action, info
```

## Benchmarks Supported

- **WebArena**: Realistic web navigation tasks
- **WorkArena**: Enterprise software tasks  
- **MiniWoB**: Micro web tasks
- **WebShop**: E-commerce navigation
- **Mind2Web**: Real-world web interactions
- **GAIA**: General AI Assistant benchmark for complex, multi-step tasks

## Performance Tips

1. **Use Vision Selectively**: Disable vision for simple tasks to improve speed
2. **Optimize Viewport**: Use smaller viewports for MiniWoB tasks
3. **Batch Execution**: Use AgentLab's parallel execution for multiple tasks
4. **Memory Management**: Disable memory for short tasks

### Parallel Execution

AgentLab supports parallel execution of experiments. Here are different ways to configure it:

```python
import multiprocessing

# Method 1: Pass n_jobs to study.run()
study.run(n_jobs=5)  # Run 5 tasks in parallel
study.run(n_jobs=-1)  # Use all available cores

# Method 2: Pass n_jobs to make_study() (if supported)
study = make_study(
    benchmark="webarena",
    agent_args=[agent_args],
    n_jobs=multiprocessing.cpu_count() - 1
)

# Method 3: Sequential execution for debugging
study.run(n_jobs=1)  # Run tasks one at a time
```

**Recommended values**:
- For I/O-bound tasks (most web tasks): 10-50 parallel jobs
- For debugging: n_jobs=1
- For maximum throughput: n_jobs=-1 or n_jobs=multiprocessing.cpu_count()

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure AgentLab is installed: `pip install agentlab`
2. **Browser Issues**: Check Playwright installation: `playwright install chromium`
3. **API Keys**: Set environment variables for your LLM provider

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug info
agent = BrowserUseAgent(agent_args)
```

## Contributing

To contribute to the AgentLab integration:

1. Fork the repository
2. Create a feature branch
3. Add tests in `tests/test_agentlab_integration.py`
4. Submit a pull request

## References

- [AgentLab Documentation](https://github.com/ServiceNow/AgentLab)
- [BrowserGym Documentation](https://github.com/ServiceNow/BrowserGym)
- [Browser-Use Documentation](https://github.com/gregpr07/browser-use)