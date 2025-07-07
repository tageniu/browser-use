# =========================================================================================================================
# source .venv/bin/activate
# Apple Trade-in
# python eval/webarena_service.py --model gpt-o4-mini --eval-model gpt-o4-mini --start 812 --end 813 --max-steps 15 --max-retries 1
# python eval/webarena_service.py --model gpt-4.1 --eval-model gpt-4.1 --start 812 --end 813 --max-steps 15 --max-retries 1
# =========================================================================================================================
# python eval/webarena_service.py --model gpt-4.1 --eval-model gpt-4.1 --start 113 --end 114 --max-steps 15 --max-retries 2
# python eval/webarena_service.py --model gpt-4.1 --eval-model gpt-4.1 --start 114 --end 115 --max-steps 15 --max-retries 2

# Reddit
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --start 69 --end 70 --max-steps 15

# Shopping 1
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --start 26 --end 27 --max-steps 15

# Shopping 2
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --start 161 --end 162 --max-steps 15

# Shopping admin
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --start 198 --end 199 --max-steps 15

# Gitlab
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --start 667 --end 668 --max-steps 15

# -- Batch testing --
# Reddit
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --max-parallel 10 --start 401 --end 411 --max-steps 15 --max-retries 1
# python eval/webarena_service.py --model gpt-4.1 --eval-model gpt-4.1 --max-parallel 10 --start 401 --end 411 --max-steps 15 --max-retries 1
# Shopping
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --max-parallel 10 --start 141 --end 151 --max-steps 15 --max-retries 1
# python eval/webarena_service.py --model gpt-4.1 --eval-model gpt-4.1 --max-parallel 10 --start 141 --end 151 --max-steps 15 --max-retries 1
# Shopping admin
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --max-parallel 10 --start 111 --end 121 --max-steps 15 --max-retries 1
# python eval/webarena_service.py --model gpt-4.1 --eval-model gpt-4.1 --max-parallel 10 --start 111 --end 121 --max-steps 15 --max-retries 1
# python eval/webarena_service.py --model gpt-o4-mini --eval-model gpt-o4-mini --max-parallel 10 --start 111 --end 121 --max-steps 15 --max-retries 1
# Gitlab
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --max-parallel 10 --start 303 --end 313 --max-steps 15 --max-retries 1
# python eval/webarena_service.py --model gpt-4.1 --eval-model gpt-4.1 --max-parallel 10 --start 303 --end 313 --max-steps 15 --max-retries 1
# python eval/webarena_service.py --model gpt-o4-mini --eval-model gpt-o4-mini --max-parallel 10 --start 303 --end 313 --max-steps 15 --max-retries 1
# ============================================================================================================================================

import asyncio
import json
import logging
import sys
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import anyio
from dotenv import load_dotenv
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import SystemMessage, UserMessage
from browser_use.agent.service import Agent
from browser_use.browser import BrowserSession, BrowserProfile
from browser_use.agent.views import AgentHistoryList
from browser_use.agent.memory import MemoryConfig

class StreamToLogger:
    """Redirect stdout/stderr to logger"""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
    
    def flush(self):
        pass

# Configure logging
def setup_logging():
	# Create logs directory if it doesn't exist
	logs_dir = Path("logs")
	logs_dir.mkdir(exist_ok=True)
	
	# Create a timestamp for the log file
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	log_file = logs_dir / f"webarena_{timestamp}.log"
	
	# Configure logging format
	log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	
	# Configure file handler
	file_handler = logging.FileHandler(log_file)
	file_handler.setFormatter(logging.Formatter(log_format))
	
	# Configure root logger
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.INFO)
	root_logger.addHandler(file_handler)
	
	# Redirect stdout and stderr to logger
	sys.stdout = StreamToLogger(logging.getLogger('stdout'), logging.INFO)
	sys.stderr = StreamToLogger(logging.getLogger('stderr'), logging.ERROR)
	
	return log_file

# Initialize logging
log_file = setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_file}")

SUPPORTED_MODELS = {
	# Anthropic
	'claude-3.5-sonnet': {
		'provider': 'anthropic',
		'model_name': 'claude-3-5-sonnet-20240620',
		'api_key_env': 'ANTHROPIC_API_KEY',
	},
	'claude-3.5-sonnet-exp': {
		'provider': 'anthropic',
		'model_name': 'claude-3-5-sonnet-20241022',
		'api_key_env': 'ANTHROPIC_API_KEY',
	},
	'claude-3.7-sonnet-exp': {
		'provider': 'anthropic',
		'model_name': 'claude-3-7-sonnet-20250219',
		'api_key_env': 'ANTHROPIC_API_KEY',
	},
	'claude-sonnet-4': {
		'provider': 'anthropic',
		'model_name': 'claude-sonnet-4-20250514',
		'api_key_env': 'ANTHROPIC_API_KEY',
	},
	'claude-opus-4': {
		'provider': 'anthropic',
		'model_name': 'claude-opus-4-20250514',
		'api_key_env': 'ANTHROPIC_API_KEY',
	},
	# Deepseek (via OpenAI Compatible API)
	'deepseek-reasoner': {
		'provider': 'openai_compatible',
		'model_name': 'deepseek-reasoner',
		'base_url': 'https://api.deepseek.com/v1',
		'api_key_env': 'DEEPSEEK_API_KEY',
	},
	'deepseek-chat': {
		'provider': 'openai_compatible',
		'model_name': 'deepseek-chat',
		'base_url': 'https://api.deepseek.com/v1',
		'api_key_env': 'DEEPSEEK_API_KEY',
	},
	# Google
	'gemini-1.5-flash': {'provider': 'google', 'model_name': 'gemini-1.5-flash-latest', 'api_key_env': 'GEMINI_API_KEY'},
	'gemini-2.0-flash-lite': {'provider': 'google', 'model_name': 'gemini-2.0-flash-lite', 'api_key_env': 'GEMINI_API_KEY'},
	'gemini-2.0-flash': {'provider': 'google', 'model_name': 'gemini-2.0-flash', 'api_key_env': 'GEMINI_API_KEY'},
	'gemini-2.5-pro': {'provider': 'google', 'model_name': 'gemini-2.5-pro-preview-03-25', 'api_key_env': 'GEMINI_API_KEY'},
	'gemini-2.5-pro-preview-05-06': {
		'provider': 'google',
		'model_name': 'gemini-2.5-pro-preview-05-06',
		'api_key_env': 'GEMINI_API_KEY',
	},
	'gemini-2.5-flash-preview': {
		'provider': 'google',
		'model_name': 'gemini-2.5-flash-preview-04-17',
		'api_key_env': 'GEMINI_API_KEY',
	},
	# OpenAI
	'gpt-4.1': {'provider': 'openai', 'model_name': 'gpt-4.1-2025-04-14', 'api_key_env': 'OPENAI_API_KEY'},
	'gpt-4.1-mini': {'provider': 'openai', 'model_name': 'gpt-4.1-mini-2025-04-14', 'api_key_env': 'OPENAI_API_KEY'},
	'gpt-4.1-nano': {'provider': 'openai', 'model_name': 'gpt-4.1-nano-2025-04-14', 'api_key_env': 'OPENAI_API_KEY'},
	'gpt-4o': {'provider': 'openai', 'model_name': 'gpt-4o', 'api_key_env': 'OPENAI_API_KEY'},
	'gpt-4o-mini': {'provider': 'openai', 'model_name': 'gpt-4o-mini', 'api_key_env': 'OPENAI_API_KEY'},
    'gpt-o1': {'provider': 'openai', 'model_name': 'o1', 'api_key_env': 'OPENAI_API_KEY'},
    'gpt-o3': {'provider': 'openai', 'model_name': 'o3', 'api_key_env': 'OPENAI_API_KEY'},
	'gpt-o4-mini': {'provider': 'openai', 'model_name': 'o4-mini', 'api_key_env': 'OPENAI_API_KEY'},
	# X.ai (via OpenAI Compatible API)
	'grok-2': {
		'provider': 'openai_compatible',
		'model_name': 'grok-2-1212',
		'base_url': 'https://api.x.ai/v1',
		'api_key_env': 'XAI_API_KEY',
	},
	'grok-3': {
		'provider': 'openai_compatible',
		'model_name': 'grok-3-beta',
		'base_url': 'https://api.x.ai/v1',
		'api_key_env': 'XAI_API_KEY',
	},
	# Groq
	'gemma2-9b-it': {
		'provider': 'openai_compatible',
		'model_name': 'gemma2-9b-it',
		'base_url': 'https://api.groq.com/openai/v1',
		'api_key_env': 'GROQ_API_KEY',
	},
	'llama-3.3-70b-versatile': {
		'provider': 'openai_compatible',
		'model_name': 'llama-3.3-70b-versatile',
		'base_url': 'https://api.groq.com/openai/v1',
		'api_key_env': 'GROQ_API_KEY',
	},
	'llama-3.1-8b-instant': {
		'provider': 'openai_compatible',
		'model_name': 'llama-3.1-8b-instant',
		'base_url': 'https://api.groq.com/openai/v1',
		'api_key_env': 'GROQ_API_KEY',
	},
	'llama3-70b-8192': {
		'provider': 'openai_compatible',
		'model_name': 'llama3-70b-8192',
		'base_url': 'https://api.groq.com/openai/v1',
		'api_key_env': 'GROQ_API_KEY',
	},
	'llama3-8b-8192': {
		'provider': 'openai_compatible',
		'model_name': 'llama3-8b-8192',
		'base_url': 'https://api.groq.com/openai/v1',
		'api_key_env': 'GROQ_API_KEY',
	},
	# Groq Preview
	'llama-4-maverick': {
		'provider': 'openai_compatible',
		'model_name': 'meta-llama/llama-4-maverick-17b-128e-instruct',
		'base_url': 'https://api.groq.com/openai/v1',
		'api_key_env': 'GROQ_API_KEY',
	},
	'llama-4-scout': {
		'provider': 'openai_compatible',
		'model_name': 'meta-llama/llama-4-scout-17b-16e-instruct',
		'base_url': 'https://api.groq.com/openai/v1',
		'api_key_env': 'GROQ_API_KEY',
	},
	# Qwen (via OpenAI Compatible API)
	'qwen-1.5-72b': {
		'provider': 'openai_compatible',
		'model_name': 'qwen-1.5-72b',
		'base_url': 'https://api.qwen.ai/v1',
		'api_key_env': 'QWEN_API_KEY',
	},
	'qwen-1.5-14b': {
		'provider': 'openai_compatible',
		'model_name': 'qwen-1.5-14b',
		'base_url': 'https://api.qwen.ai/v1',
		'api_key_env': 'QWEN_API_KEY',
	},
	'qwen-1.5-7b': {
		'provider': 'openai_compatible',
		'model_name': 'qwen-1.5-7b',
		'base_url': 'https://api.qwen.ai/v1',
		'api_key_env': 'QWEN_API_KEY',
	},
    # 'qwen': {
    #     'provider': 'openai_compatible',
    #     'model_name': 'qwen/qwen3-235b-a22b:free',
    #     'base_url': 'https://openrouter.ai/api/v1',
    #     'api_key_env': 'QWEN_API_KEY',
    # },
    'qwen': {
        'provider': 'openai_compatible',
        'model_name': 'qwen-max',
        'base_url': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
        'api_key_env': 'QWEN_API_KEY',
    },
}

@dataclass
class WebArenaTask:
    """WebArena task structure"""
    task_id: str
    intent: str
    start_url: str
    sites: List[str]
    require_login: bool = False
    storage_state: Optional[str] = None
    eval_criteria: Optional[Dict] = None
    
    def __str__(self):
        return f"WebArenaTask(id={self.task_id}, sites={self.sites}, intent='{self.intent[:50]}...')"
    
    def __repr__(self):
        return self.__str__()


class WebArenaEvaluator:
    """WebArena evaluation logic"""
    
    def __init__(self, model: BaseChatModel):
        self.model = model
    
    async def evaluate_task_completion(self, task: WebArenaTask, agent_history: AgentHistoryList) -> Dict:
        """
        Evaluate if a WebArena task was completed successfully based on functional correctness.
        """
        
        # Extract the final state information
        final_page_content = self._extract_final_page_content(agent_history)
        action_sequence = self._extract_action_sequence(agent_history)

        # Log the final page content and action sequence
        logger.info(f"Final page content: \n{final_page_content}")
        logger.info(f"Action sequence: \n{action_sequence}")
        
        # WebArena uses functional correctness evaluation
        evaluation_prompt = self._build_evaluation_prompt(task, final_page_content, action_sequence)
        
        try:
            # Get evaluation from LLM
            messages = [
                SystemMessage(content=self._get_evaluation_system_prompt()),
                UserMessage(content=evaluation_prompt)
            ]
            
            response = await self.model.ainvoke(messages)
            evaluation_result = self._parse_evaluation_response(response.completion)
            
            return {
                'task_id': task.task_id,
                'success': evaluation_result.get('success', False),
                'score': 1.0 if evaluation_result.get('success', False) else 0.0,
                'reasoning': evaluation_result.get('reasoning', ''),
                'functional_correctness': evaluation_result.get('functional_correctness', False),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}")
            return {
                'task_id': task.task_id,
                'success': False,
                'score': 0.0,
                'reasoning': f'Evaluation error: {str(e)}',
                'functional_correctness': False,
                'error': str(e)
            }
    
    def _get_evaluation_system_prompt(self) -> str:
        """System prompt for WebArena evaluation"""
        return """You are an expert evaluator for WebArena tasks. Your job is to determine if a web agent successfully completed a given task based on functional correctness.

WebArena evaluation criteria:
1. Task completion: Did the agent achieve the stated objective?
2. Functional correctness: Are the results accurate and correct?
3. Proper navigation: Did the agent navigate to the correct pages/sites?
4. Data integrity: Were any data modifications done correctly?

Evaluate based on:
- The final state of the webpage
- The sequence of actions taken
- Whether the objective was functionally achieved
- Any specific evaluation criteria provided

Respond in JSON format:
{
    "success": true/false,
    "functional_correctness": true/false,
    "reasoning": "Detailed explanation of your evaluation"
}"""

    def _build_evaluation_prompt(self, task: WebArenaTask, final_content: str, actions: List[str]) -> str:
        """Build the evaluation prompt for a specific task"""
        prompt = f"""Task to evaluate:
Intent: {task.intent}
Target sites: {', '.join(task.sites)}
Start URL: {task.start_url}

Agent's action sequence:
{chr(10).join(actions[-10:])}  # Last 10 actions

Final page content (relevant excerpts):
{final_content[:2000]}  # First 2000 chars

Evaluate whether the task was completed successfully with functional correctness."""

        # Add evaluation criteria information if available
        if task.eval_criteria:
            prompt += "\n\nEvaluation Criteria:"
            
            # Add evaluation types
            if 'eval_types' in task.eval_criteria and task.eval_criteria['eval_types']:
                prompt += f"\nEvaluation Types: {', '.join(task.eval_criteria['eval_types'])}"
            
            # Add reference answers
            if 'reference_answers' in task.eval_criteria and task.eval_criteria['reference_answers']:
                prompt += f"\nReference Answer: {task.eval_criteria['reference_answers'].get('exact_match', 'Not specified')}"
            
            # Add raw annotation
            if 'reference_answer_raw_annotation' in task.eval_criteria and task.eval_criteria['reference_answer_raw_annotation']:
                prompt += f"\nRaw Annotation: {task.eval_criteria['reference_answer_raw_annotation']}"
            
            # Add reference URL if available
            if 'reference_url' in task.eval_criteria and task.eval_criteria['reference_url']:
                prompt += f"\nReference URL: {task.eval_criteria['reference_url']}"

            # Add URL note if available
            if 'url_note' in task.eval_criteria and task.eval_criteria['url_note']:
                prompt += f"\nURL Note: {task.eval_criteria['url_note']}"

        return prompt

    def _extract_final_page_content(self, agent_history: AgentHistoryList) -> str:
        """Extract relevant content from the final page state"""
        if not agent_history.history:
            return "No history available"
            
        final_step = agent_history.history[-1]
        
        # Extract text content from the final observation
        if hasattr(final_step, 'result') and final_step.result:
            # Result is a list, so get the last result
            if isinstance(final_step.result, list) and len(final_step.result) > 0:
                last_result = final_step.result[-1]
                if hasattr(last_result, 'extracted_content') and last_result.extracted_content:
                    return str(last_result.extracted_content)
            # Handle case where result might be a single ActionResult (not a list)
            elif not isinstance(final_step.result, list) and hasattr(final_step.result, 'extracted_content') and final_step.result.extracted_content:
                return str(final_step.result.extracted_content)
        
        return "No final content available"

    def _extract_action_sequence(self, agent_history: AgentHistoryList) -> List[str]:
        """Extract the sequence of actions taken by the agent"""
        actions = []
        
        for step in agent_history.history:
            if hasattr(step, 'model_output') and step.model_output:
                if hasattr(step.model_output, 'action') and step.model_output.action:
                    # Actions are stored as a list
                    if isinstance(step.model_output.action, list):
                        for action in step.model_output.action:
                            action_str = self._format_action(action)
                            if action_str:
                                actions.append(action_str)
                    else:
                        action_str = self._format_action(step.model_output.action)
                        if action_str:
                            actions.append(action_str)
        
        return actions
    
    def _format_action(self, action) -> str:
        """Format a single action into a readable string"""
        if not action:
            return ""
            
        # Handle dictionary-style actions
        if isinstance(action, dict):
            for action_type, action_data in action.items():
                if action_data is not None:
                    if isinstance(action_data, dict):
                        # Extract relevant info from action data
                        if 'url' in action_data:
                            return f"{action_type}: {action_data['url']}"
                        elif 'index' in action_data:
                            return f"{action_type}: index {action_data['index']}"
                        elif 'text' in action_data:
                            return f"{action_type}: {action_data['text']}"
                        else:
                            return f"{action_type}: {action_data}"
                    else:
                        return f"{action_type}: {action_data}"
        
        # Handle object-style actions  
        if hasattr(action, '__dict__'):
            action_dict = action.__dict__ if hasattr(action, '__dict__') else {}
            non_null_actions = {k: v for k, v in action_dict.items() if v is not None}
            if non_null_actions:
                action_type = list(non_null_actions.keys())[0]
                action_data = non_null_actions[action_type]
                if isinstance(action_data, dict):
                    if 'url' in action_data:
                        return f"{action_type}: {action_data['url']}"
                    elif 'index' in action_data:
                        return f"{action_type}: index {action_data['index']}"
                    elif 'text' in action_data:
                        return f"{action_type}: {action_data['text']}"
                    else:
                        return f"{action_type}: {action_data}"
                else:
                    return f"{action_type}: {action_data}"
        
        return str(action)

    def _parse_evaluation_response(self, response: str) -> Dict:
        """Parse the LLM evaluation response"""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                # Fallback parsing
                success = 'true' in response.lower() and 'success' in response.lower()
                return {
                    'success': success,
                    'functional_correctness': success,
                    'reasoning': response
                }
                
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            success = 'success' in response.lower() or 'completed' in response.lower()
            return {
                'success': success,
                'functional_correctness': success,
                'reasoning': response
            }

def load_webarena_tasks(config_dir: str = "config_files/webarena") -> List[WebArenaTask]:
    """Load WebArena tasks from configuration files in numerical order (starting from 0.json)"""
    tasks = []
    config_path = Path(config_dir)

    # Get all json files and sort them numerically
    config_files = sorted(
        config_path.glob("*.json"),
        key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf')
    )

    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                task_config = json.load(f)
                
            task = WebArenaTask(
                task_id=task_config.get('task_id', config_file.stem),
                intent=task_config.get('intent', ''),
                start_url=task_config.get('start_url', ''),
                sites=task_config.get('sites', []),
                require_login=task_config.get('require_login', False),
                storage_state=task_config.get('storage_state'),
                eval_criteria=task_config.get('eval'),
            )
            tasks.append(task)
            
        except Exception as e:
            logger.error(f"Error loading task from {config_file}: {e}")
    
    return tasks

class WebArenaTaskRunner:
    """Handles running WebArena tasks with browser-use agents"""
    
    def __init__(self, 
                 llm: BaseChatModel,
                 evaluator: WebArenaEvaluator,
                 headless: bool = True,
                 max_steps: int = 20):
        self.llm = llm
        self.evaluator = evaluator
        self.headless = headless
        self.max_steps = max_steps
        self.active_sessions = {}  # Track active browser sessions
    
    async def _load_previous_attempts(self, task_id: str) -> List[Dict]:
        """Load previous attempt histories for a given task"""
        previous_attempts = []
        
        # Check for previous history files
        history_dir = Path("saved_trajectories/webarena/agent_history")
        results_dir = Path("saved_trajectories/webarena")
        
        if not history_dir.exists():
            logger.info(f"No history directory found for task {task_id}")
            return previous_attempts
            
        # Look for history files for this task (including retry attempts)
        history_files = list(history_dir.glob(f"{task_id}_history*.json"))
        logger.info(f"Found {len(history_files)} history files for task {task_id}: {[f.name for f in history_files]}")
        
        for history_file in history_files:
            try:
                logger.info(f"Loading history file: {history_file}")
                async with await anyio.open_file(history_file, 'r') as f:
                    history_data = json.loads(await f.read())
                    
                # Extract key information from the history
                attempt_info = {
                    'file': history_file.name,
                    'steps': len(history_data.get('history', [])),
                    'actions': [],
                    'memory_entries': [],
                    'thinking_entries': [],
                    'evaluation_entries': [],
                    'final_state': None,
                    'errors': [],
                    'urls_visited': [],
                    'final_result': None,
                    'evaluator_feedback': None
                }
                
                logger.info(f"Processing {attempt_info['steps']} steps from {history_file.name}")
                
                # Extract actions and other important info from each step
                for step_idx, step in enumerate(history_data.get('history', [])):
                    logger.debug(f"Processing step {step_idx + 1}/{attempt_info['steps']}")
                    
                    # Extract actions
                    if step.get('model_output') and step['model_output'].get('action'):
                        for action in step['model_output']['action']:
                            if action:
                                action_str = self._format_action_from_dict(action)
                                if action_str:
                                    attempt_info['actions'].append(action_str)
                    
                    # Extract memory entries
                    if step.get('model_output') and step['model_output'].get('memory'):
                        memory = step['model_output']['memory']
                        if memory and memory.strip():
                            attempt_info['memory_entries'].append({
                                'step': step_idx + 1,
                                'memory': memory
                            })
                            logger.debug(f"Step {step_idx + 1} memory: {memory[:100]}...")
                    
                    # Extract thinking entries
                    if step.get('model_output') and step['model_output'].get('thinking'):
                        thinking = step['model_output']['thinking']
                        if thinking and thinking.strip():
                            attempt_info['thinking_entries'].append({
                                'step': step_idx + 1,
                                'thinking': thinking
                            })
                            logger.debug(f"Step {step_idx + 1} thinking: {thinking[:100]}...")
                    
                    # Extract evaluation entries
                    if step.get('model_output') and step['model_output'].get('evaluation_previous_goal'):
                        evaluation = step['model_output']['evaluation_previous_goal']
                        if evaluation and evaluation.strip():
                            attempt_info['evaluation_entries'].append({
                                'step': step_idx + 1,
                                'evaluation': evaluation
                            })
                            logger.debug(f"Step {step_idx + 1} evaluation: {evaluation[:100]}...")
                    
                    # Extract URLs visited
                    if step.get('state') and step['state'].get('url'):
                        url = step['state']['url']
                        if url and url not in attempt_info['urls_visited']:
                            attempt_info['urls_visited'].append(url)
                    
                    # Check for errors in results
                    if step.get('result'):
                        for result in step['result']:
                            if result.get('error'):
                                attempt_info['errors'].append({
                                    'step': step_idx + 1,
                                    'error': result['error']
                                })
                                logger.debug(f"Step {step_idx + 1} error: {result['error']}")
                            
                            # Extract final result if this is a done action
                            if result.get('is_done') and result.get('extracted_content'):
                                attempt_info['final_result'] = result['extracted_content']
                                logger.info(f"Found final result in step {step_idx + 1}: {result['extracted_content'][:200]}...")
                
                # Get final state
                if history_data.get('history'):
                    final_step = history_data['history'][-1]
                    if final_step.get('state'):
                        attempt_info['final_state'] = {
                            'url': final_step['state'].get('url'),
                            'title': final_step['state'].get('title')
                        }
                        logger.info(f"Final state: {attempt_info['final_state']}")
                
                # Load evaluator feedback from result file
                result_file = results_dir / f"{task_id}.json"
                if result_file.exists():
                    try:
                        async with await anyio.open_file(result_file, 'r') as f:
                            result_data = json.loads(await f.read())
                        
                        # Extract evaluation information
                        if 'evaluation' in result_data:
                            evaluation_data = result_data['evaluation']
                            attempt_info['evaluator_feedback'] = {
                                'success': evaluation_data.get('success', False),
                                'score': evaluation_data.get('score', 0.0),
                                'reasoning': evaluation_data.get('reasoning', ''),
                                'functional_correctness': evaluation_data.get('functional_correctness', False),
                                'error': evaluation_data.get('error')
                            }
                            logger.info(f"Loaded evaluator feedback: success={evaluation_data.get('success')}, score={evaluation_data.get('score')}")
                            logger.debug(f"Evaluator reasoning: {evaluation_data.get('reasoning', '')[:200]}...")
                        else:
                            logger.warning(f"No evaluation data found in result file for task {task_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load evaluator feedback from result file {result_file}: {e}")
                else:
                    logger.info(f"No result file found for task {task_id}")
                
                # Log summary statistics
                logger.info(f"Attempt summary for {history_file.name}:")
                logger.info(f"  • Steps: {attempt_info['steps']}")
                logger.info(f"  • Actions: {len(attempt_info['actions'])}")
                logger.info(f"  • Memory entries: {len(attempt_info['memory_entries'])}")
                logger.info(f"  • Thinking entries: {len(attempt_info['thinking_entries'])}")
                logger.info(f"  • Evaluation entries: {len(attempt_info['evaluation_entries'])}")
                logger.info(f"  • URLs visited: {len(attempt_info['urls_visited'])}")
                logger.info(f"  • Errors: {len(attempt_info['errors'])}")
                logger.info(f"  • Final result: {'Yes' if attempt_info['final_result'] else 'No'}")
                logger.info(f"  • Evaluator feedback: {'Yes' if attempt_info['evaluator_feedback'] else 'No'}")
                
                previous_attempts.append(attempt_info)
                
            except Exception as e:
                logger.error(f"Failed to load history file {history_file}: {e}", exc_info=True)
        
        logger.info(f"Successfully loaded {len(previous_attempts)} previous attempts for task {task_id}")
        return previous_attempts
    
    def _format_action_from_dict(self, action_dict: Dict) -> str:
        """Format action dictionary into readable string"""
        if not action_dict:
            return ""
            
        for action_type, action_data in action_dict.items():
            if action_data is not None:
                if isinstance(action_data, dict):
                    if 'url' in action_data:
                        return f"{action_type}: {action_data['url']}"
                    elif 'index' in action_data:
                        return f"{action_type}: index {action_data['index']}"
                    elif 'text' in action_data:
                        return f"{action_type}: {action_data['text']}"
                    else:
                        return f"{action_type}: {action_data}"
                else:
                    return f"{action_type}: {action_data}"
        
        return str(action_dict)
    
    def _build_retry_context(self, task: WebArenaTask, previous_attempts: List[Dict], retry_info: Dict) -> str:
        """Build context from previous attempts to help guide the retry"""
        if not previous_attempts:
            logger.info("No previous attempts found, skipping retry context generation")
            return ""
        
        logger.info(f"Building retry context for {len(previous_attempts)} previous attempts")
        
        context_parts = [
            f"⚠️  RETRY ATTEMPT {retry_info.get('retry_attempt', 1)}: This is a retry of a failed task.",
            f"Previous failure reason: {retry_info.get('original_failure_reason', 'Unknown')}",
            "",
            "📋 PREVIOUS ATTEMPTS SUMMARY:"
        ]
        
        # Analyze patterns across attempts
        stuck_patterns = []
        common_actions = []
        final_urls = []
        memory_patterns = []
        thinking_patterns = []
        evaluation_patterns = []
        evaluator_feedback_patterns = []
        
        for i, attempt in enumerate(previous_attempts, 1):
            logger.info(f"Analyzing attempt {i}: {attempt['file']}")
            
            context_parts.append(f"\nAttempt {i} ({attempt['file']}):")
            context_parts.append(f"  • Steps taken: {attempt['steps']}")
            
            if attempt['final_state']:
                final_url = attempt['final_state']['url']
                final_urls.append(final_url)
                context_parts.append(f"  • Final state: {final_url}")
                logger.debug(f"Attempt {i} final URL: {final_url}")
            
            if attempt['errors']:
                error_summary = f"  • Errors encountered: {len(attempt['errors'])} errors"
                context_parts.append(error_summary)
                logger.debug(f"Attempt {i} errors: {[e['error'][:50] + '...' for e in attempt['errors'][:3]]}")
            
            # Show last few actions to indicate where it got stuck
            if attempt['actions']:
                last_actions = attempt['actions'][-5:]  # Last 5 actions
                context_parts.append(f"  • Last actions: {' → '.join(last_actions)}")
                logger.debug(f"Attempt {i} last actions: {last_actions}")
                
                # Track common actions for pattern analysis
                common_actions.extend(last_actions)
                
                # Check for stuck patterns
                if len(last_actions) >= 3:
                    # Check if stuck in customer listing (common pattern)
                    if any('customer' in action.lower() for action in last_actions[-3:]):
                        stuck_patterns.append('customer_listing')
                        logger.info(f"Attempt {i}: Detected customer listing pattern")
                    # Check if stuck in search loops
                    if any('search' in action.lower() for action in last_actions[-3:]):
                        stuck_patterns.append('search_loops')
                        logger.info(f"Attempt {i}: Detected search loop pattern")
                    # Check if stuck clicking same elements
                    if len(set(last_actions[-3:])) <= 1:
                        stuck_patterns.append('repetitive_clicking')
                        logger.info(f"Attempt {i}: Detected repetitive clicking pattern")
            
            # Analyze memory patterns
            if attempt['memory_entries']:
                context_parts.append(f"  • Memory entries: {len(attempt['memory_entries'])}")
                # Extract key memory insights
                for mem_entry in attempt['memory_entries'][-3:]:  # Last 3 memory entries
                    memory_text = mem_entry['memory'].lower()
                    if 'stuck' in memory_text or 'not found' in memory_text:
                        memory_patterns.append('stuck_in_search')
                        logger.debug(f"Attempt {i} memory indicates being stuck: {mem_entry['memory'][:100]}...")
                    elif 'customer' in memory_text:
                        memory_patterns.append('customer_focus')
                        logger.debug(f"Attempt {i} memory focuses on customers: {mem_entry['memory'][:100]}...")
            
            # Analyze thinking patterns
            if attempt['thinking_entries']:
                context_parts.append(f"  • Thinking entries: {len(attempt['thinking_entries'])}")
                # Extract key thinking insights
                for think_entry in attempt['thinking_entries'][-2:]:  # Last 2 thinking entries
                    thinking_text = think_entry['thinking'].lower()
                    if 'need to try' in thinking_text or 'alternative' in thinking_text:
                        thinking_patterns.append('seeking_alternatives')
                        logger.debug(f"Attempt {i} thinking shows seeking alternatives: {think_entry['thinking'][:100]}...")
                    elif 'stuck' in thinking_text or 'not working' in thinking_text:
                        thinking_patterns.append('recognizing_failure')
                        logger.debug(f"Attempt {i} thinking shows recognizing failure: {think_entry['thinking'][:100]}...")
            
            # Analyze evaluation patterns
            if attempt['evaluation_entries']:
                context_parts.append(f"  • Evaluation entries: {len(attempt['evaluation_entries'])}")
                # Extract key evaluation insights
                for eval_entry in attempt['evaluation_entries'][-2:]:  # Last 2 evaluation entries
                    eval_text = eval_entry['evaluation'].lower()
                    if 'partial' in eval_text or 'not complete' in eval_text:
                        evaluation_patterns.append('partial_success')
                        logger.debug(f"Attempt {i} evaluation shows partial success: {eval_entry['evaluation'][:100]}...")
                    elif 'failed' in eval_text or 'unsuccessful' in eval_text:
                        evaluation_patterns.append('recognized_failure')
                        logger.debug(f"Attempt {i} evaluation shows recognized failure: {eval_entry['evaluation'][:100]}...")
            
            # Analyze evaluator feedback
            if attempt['evaluator_feedback']:
                feedback = attempt['evaluator_feedback']
                context_parts.append(f"  • Evaluator feedback: Success={feedback['success']}, Score={feedback['score']}")
                logger.info(f"Attempt {i} evaluator feedback: success={feedback['success']}, score={feedback['score']}")
                
                # Extract key insights from evaluator reasoning
                reasoning_text = feedback['reasoning'].lower()
                if 'wrong page' in reasoning_text or 'incorrect page' in reasoning_text:
                    evaluator_feedback_patterns.append('wrong_page')
                    logger.debug(f"Attempt {i} evaluator indicates wrong page: {feedback['reasoning'][:100]}...")
                elif 'not found' in reasoning_text or 'missing' in reasoning_text:
                    evaluator_feedback_patterns.append('information_not_found')
                    logger.debug(f"Attempt {i} evaluator indicates information not found: {feedback['reasoning'][:100]}...")
                elif 'incomplete' in reasoning_text or 'partial' in reasoning_text:
                    evaluator_feedback_patterns.append('incomplete_task')
                    logger.debug(f"Attempt {i} evaluator indicates incomplete task: {feedback['reasoning'][:100]}...")
                elif 'navigation' in reasoning_text or 'navigation failed' in reasoning_text:
                    evaluator_feedback_patterns.append('navigation_failure')
                    logger.debug(f"Attempt {i} evaluator indicates navigation failure: {feedback['reasoning'][:100]}...")
                
                # Show evaluator reasoning
                context_parts.append(f"  • Evaluator reasoning: {feedback['reasoning'][:200]}...")
                logger.debug(f"Attempt {i} full evaluator reasoning: {feedback['reasoning']}")
            
            # Show URLs visited
            if attempt['urls_visited']:
                context_parts.append(f"  • URLs visited: {len(attempt['urls_visited'])} unique pages")
                logger.debug(f"Attempt {i} URLs: {attempt['urls_visited']}")
            
            # Show final result if available
            if attempt['final_result']:
                context_parts.append(f"  • Final result: {attempt['final_result'][:100]}...")
                logger.info(f"Attempt {i} final result: {attempt['final_result'][:200]}...")
        
        # Add pattern analysis and specific guidance
        context_parts.extend([
            "",
            "🔍 PATTERN ANALYSIS:"
        ])
        
        if stuck_patterns:
            unique_patterns = list(set(stuck_patterns))
            context_parts.append(f"  • Detected stuck patterns: {', '.join(unique_patterns)}")
            logger.info(f"Detected stuck patterns: {unique_patterns}")
            
            # Provide specific guidance based on patterns
            if 'customer_listing' in unique_patterns:
                context_parts.extend([
                    "  • Customer listing pattern detected: Previous attempts got stuck in customer management sections",
                    "    → Try exploring other sections first (Sales, Reports, Catalog, etc.)",
                    "    → Look for review/feedback sections outside of customer management",
                    "    → Consider searching for the product directly in different contexts"
                ])
            
            if 'search_loops' in unique_patterns:
                context_parts.extend([
                    "  • Search loop pattern detected: Previous attempts got stuck in search operations",
                    "    → Try direct navigation to specific sections instead of searching",
                    "    → Look for menu items or navigation that might lead to the target information",
                    "    → Consider browsing through different admin sections systematically"
                ])
            
            if 'repetitive_clicking' in unique_patterns:
                context_parts.extend([
                    "  • Repetitive clicking pattern detected: Previous attempts kept clicking the same elements",
                    "    → The current approach is not working, try a completely different navigation path",
                    "    → Look for alternative ways to access the same information",
                    "    → Consider if the information might be in a different section entirely"
                ])
        
        # Analyze memory and thinking patterns
        if memory_patterns:
            unique_memory_patterns = list(set(memory_patterns))
            context_parts.append(f"  • Memory patterns: {', '.join(unique_memory_patterns)}")
            logger.info(f"Detected memory patterns: {unique_memory_patterns}")
            
            if 'stuck_in_search' in unique_memory_patterns:
                context_parts.append("    → Previous attempts recognized being stuck in search - try different approaches")
            
            if 'customer_focus' in unique_memory_patterns:
                context_parts.append("    → Previous attempts focused heavily on customer sections - explore other areas")
        
        if thinking_patterns:
            unique_thinking_patterns = list(set(thinking_patterns))
            context_parts.append(f"  • Thinking patterns: {', '.join(unique_thinking_patterns)}")
            logger.info(f"Detected thinking patterns: {unique_thinking_patterns}")
            
            if 'seeking_alternatives' in unique_thinking_patterns:
                context_parts.append("    → Previous attempts were seeking alternatives - this confirms the need for different approach")
            
            if 'recognizing_failure' in unique_thinking_patterns:
                context_parts.append("    → Previous attempts recognized their approach wasn't working - avoid similar strategies")
        
        if evaluation_patterns:
            unique_eval_patterns = list(set(evaluation_patterns))
            context_parts.append(f"  • Evaluation patterns: {', '.join(unique_eval_patterns)}")
            logger.info(f"Detected evaluation patterns: {unique_eval_patterns}")
            
            if 'partial_success' in unique_eval_patterns:
                context_parts.append("    → Previous attempts had partial success - build on what worked")
            
            if 'recognized_failure' in unique_eval_patterns:
                context_parts.append("    → Previous attempts recognized complete failure - need completely different approach")
        
        # Analyze evaluator feedback patterns
        if evaluator_feedback_patterns:
            unique_feedback_patterns = list(set(evaluator_feedback_patterns))
            context_parts.append(f"  • Evaluator feedback patterns: {', '.join(unique_feedback_patterns)}")
            logger.info(f"Detected evaluator feedback patterns: {unique_feedback_patterns}")
            
            if 'wrong_page' in unique_feedback_patterns:
                context_parts.extend([
                    "    → Evaluator indicated previous attempts ended up on wrong pages",
                    "    → Focus on navigation to the correct sections and pages",
                    "    → Double-check URLs and page content to ensure you're in the right place"
                ])
            
            if 'information_not_found' in unique_feedback_patterns:
                context_parts.extend([
                    "    → Evaluator indicated the required information was not found",
                    "    → Try different search terms or explore different sections",
                    "    → Look for alternative ways the information might be presented"
                ])
            
            if 'incomplete_task' in unique_feedback_patterns:
                context_parts.extend([
                    "    → Evaluator indicated previous attempts were incomplete",
                    "    → Ensure you complete all required steps of the task",
                    "    → Double-check that you've found everything requested"
                ])
            
            if 'navigation_failure' in unique_feedback_patterns:
                context_parts.extend([
                    "    → Evaluator indicated navigation problems in previous attempts",
                    "    → Focus on proper navigation techniques and page loading",
                    "    → Ensure you're clicking the right elements"
                ])
        
        # Check if all attempts ended at similar URLs
        if len(set(final_urls)) <= 2 and len(final_urls) > 1:
            context_parts.extend([
                f"  • All attempts ended at similar locations: {', '.join(set(final_urls))}",
                "    → This suggests the information is not in this section",
                "    → Try exploring completely different sections of the admin panel"
            ])
            logger.info(f"All attempts ended at similar URLs: {set(final_urls)}")
        
        # Check if attempts ran out of steps
        max_steps_used = max(attempt['steps'] for attempt in previous_attempts)
        if max_steps_used >= 15:  # Assuming max_steps is around 15-20
            context_parts.extend([
                f"  • Previous attempts used {max_steps_used} steps (near limit)",
                "    → Be more efficient and direct in your approach",
                "    → Prioritize actions that are most likely to lead to the target information",
                "    → Avoid exploratory actions that don't directly contribute to the goal"
            ])
            logger.info(f"Previous attempts used {max_steps_used} steps (near limit)")
        
        # Add insights from memory and thinking
        if memory_patterns or thinking_patterns:
            context_parts.extend([
                "",
                "🧠 COGNITIVE INSIGHTS:"
            ])
            
            if 'stuck_in_search' in memory_patterns:
                context_parts.append("  • Previous attempts recognized being stuck in search operations")
                context_parts.append("  • This confirms the need to try different navigation strategies")
            
            if 'seeking_alternatives' in thinking_patterns:
                context_parts.append("  • Previous attempts were actively seeking alternative approaches")
                context_parts.append("  • This validates the need for a completely different strategy")
            
            if 'recognizing_failure' in thinking_patterns:
                context_parts.append("  • Previous attempts recognized their approach wasn't working")
                context_parts.append("  • Avoid similar strategies and try fundamentally different approaches")
        
        # Add evaluator feedback insights
        if evaluator_feedback_patterns:
            context_parts.extend([
                "",
                "📊 EVALUATOR FEEDBACK INSIGHTS:"
            ])
            
            # Get the most recent evaluator feedback for specific guidance
            recent_feedback = None
            for attempt in reversed(previous_attempts):
                if attempt['evaluator_feedback']:
                    recent_feedback = attempt['evaluator_feedback']
                    break
            
            if recent_feedback:
                context_parts.extend([
                    f"  • Most recent evaluator score: {recent_feedback['score']}/1.0",
                    f"  • Success status: {'Yes' if recent_feedback['success'] else 'No'}",
                    f"  • Key feedback: {recent_feedback['reasoning'][:300]}...",
                    "",
                    "  • Based on evaluator feedback:"
                ])
                
                if 'wrong_page' in evaluator_feedback_patterns:
                    context_parts.append("    → Ensure you navigate to the correct pages and sections")
                    context_parts.append("    → Verify you're on the right page before proceeding")
                
                if 'information_not_found' in evaluator_feedback_patterns:
                    context_parts.append("    → Try different search strategies and explore more sections")
                    context_parts.append("    → Look for information in unexpected places")
                
                if 'incomplete_task' in evaluator_feedback_patterns:
                    context_parts.append("    → Make sure to complete all aspects of the task")
                    context_parts.append("    → Double-check that you've found everything requested")
                
                if 'navigation_failure' in evaluator_feedback_patterns:
                    context_parts.append("    → Focus on proper navigation and page loading")
                    context_parts.append("    → Ensure you're clicking the right elements")
        
        context_parts.extend([
            "",
            "🎯 RETRY STRATEGY:",
            "• Analyze the previous attempts to understand where they got stuck",
            "• Avoid repeating the same unsuccessful approaches",
            "• Consider alternative navigation paths or search strategies",
            "• If previous attempts got stuck in a specific section, try different sections first",
            "• Look for patterns in the failures to identify better approaches",
            "• Be more systematic and efficient in your navigation",
            "• Focus on sections that are most likely to contain the target information",
            "• Use the cognitive insights from previous attempts to guide your strategy",
            "• Pay attention to evaluator feedback to understand what went wrong",
            "• Address the specific issues identified by the evaluator"
        ])
        
        final_context = "\n".join(context_parts)
        logger.info(f"Generated retry context with {len(final_context)} characters and {len(final_context.split(chr(10)))} lines")
        logger.debug(f"Retry context preview: {final_context[:500]}...")
        
        return final_context

    async def run_task(self, task: WebArenaTask, retry_info: Optional[Dict] = None) -> Dict:
        """Run a single WebArena task"""
        logger.info(f"Starting WebArena task: {task.task_id}")
        if retry_info:
            logger.info(f"Retry attempt {retry_info.get('retry_attempt', 1)} for task {task.task_id}")
        logger.info(f"Task intent: {task.intent}")
        logger.info(f"Start URL: {task.start_url}")
        logger.info(f"Target sites: {task.sites}")
        logger.info(f"Requires login: {task.require_login}")
        
        # Load previous attempts if this is a retry
        previous_attempts = []
        retry_context = ""
        if retry_info:
            logger.info(f"Loading previous attempts for retry attempt {retry_info.get('retry_attempt', 1)}")
            previous_attempts = await self._load_previous_attempts(task.task_id)
            retry_context = self._build_retry_context(task, previous_attempts, retry_info)
            logger.info(f"Loaded {len(previous_attempts)} previous attempts for task {task.task_id}")
            logger.info(f"Generated retry context: {len(retry_context)} characters")
            
            if retry_context:
                logger.info("Retry context includes:")
                logger.info(f"  • Pattern analysis: {len([line for line in retry_context.split(chr(10)) if '🔍 PATTERN ANALYSIS:' in line])} sections")
                logger.info(f"  • Cognitive insights: {len([line for line in retry_context.split(chr(10)) if '🧠 COGNITIVE INSIGHTS:' in line])} sections")
                logger.info(f"  • Strategy guidance: {len([line for line in retry_context.split(chr(10)) if '🎯 RETRY STRATEGY:' in line])} sections")
            else:
                logger.warning("No retry context generated - this may indicate no previous attempts found or analysis failed")
        else:
            logger.info("No retry info provided - this is an initial attempt")
        
        # Create browser session with WebArena configuration
        browser_profile = BrowserProfile(
            headless=self.headless,
            user_data_dir=None, #f"./browser_data/{task.task_id}",
            viewport={'width': 1280, 'height': 720},
            storage_state=task.storage_state
        )
        
        browser_session = BrowserSession(browser_profile=browser_profile)
        self.active_sessions[task.task_id] = browser_session
        
        try:
            logger.info("Starting browser session...")
            await browser_session.start()
            
            # Load storage state if required
            if task.require_login and task.storage_state:
                storage_state_path = Path(task.storage_state)
                if storage_state_path.exists():
                    logger.info(f"Loading storage state from {storage_state_path}")
                    await browser_session.load_storage_state()
                else:
                    logger.warning(f"Storage state file not found: {storage_state_path}")
            
            # Create agent with task context, including retry context if available
            agent_task = f"Navigate to {task.start_url} and {task.intent}"
            if retry_context:
                agent_task = f"{retry_context}\n\n{agent_task}"
                logger.info("Agent task includes retry context")
                logger.debug(f"Full agent task preview: {agent_task[:1000]}...")
            else:
                logger.info("Agent task does not include retry context")
            
            logger.info(f"Initializing agent with task: {agent_task[:200]}...")
            
            # Configure memory interval to prevent information loss
            memory_config = MemoryConfig(
                agent_id=f"webarena_agent_{task.task_id}",
                memory_interval=10,
                llm_instance=self.llm
            )
            
            agent = Agent(
                task=agent_task,
                llm=self.llm,
                browser_session=browser_session,
                enable_memory=True,
                memory_config=memory_config,
                max_actions_per_step=10,
                max_failures=3,
                retry_delay=10,
                validate_output=True,
                use_vision=True,
                # tool_calling_method='auto'
            )
            
            # Run the agent
            logger.info(f"Running agent for task {task.task_id}")
            agent_history = await agent.run(max_steps=self.max_steps)
            
            # Log agent execution summary
            logger.info(f"Agent execution completed for task {task.task_id}")
            logger.info(f"  • Steps taken: {len(agent_history.history)}")
            logger.info(f"  • Final success: {agent_history.is_successful()}")
            final_result = agent_history.final_result()
            logger.info(f"  • Final result: {final_result[:200] if final_result else 'None'}...")
            
            # Evaluate the result
            logger.info(f"Evaluating task {task.task_id}")
            evaluation = await self.evaluator.evaluate_task_completion(task, agent_history)
            
            # Add retry information to evaluation if this is a retry
            if retry_info:
                evaluation.update(retry_info)
                logger.info(f"Updated evaluation with retry info: attempt {retry_info.get('retry_attempt', 1)}")
            
            # Log evaluation results
            logger.info(f"Task {task.task_id} evaluation results:")
            logger.info(f"Success: {evaluation['success']}")
            logger.info(f"Score: {evaluation['score']}")
            logger.info(f"Reasoning: \n{evaluation['reasoning']}")
            
            # Save trajectory
            result_data = {
                'task_id': task.task_id,
                'task': task.intent,
                'sites': task.sites,
                'start_url': task.start_url,
                'agent_history': agent_history.model_dump() if hasattr(agent_history, 'model_dump') else str(agent_history),
                'evaluation': evaluation,
                'completed_at': datetime.now().isoformat()
            }
            
            # Add retry metadata if this is a retry
            if retry_info:
                result_data['retry_metadata'] = retry_info
                result_data['previous_attempts_summary'] = [
                    {
                        'file': attempt['file'],
                        'steps': attempt['steps'],
                        'final_state': attempt['final_state'],
                        'error_count': len(attempt['errors']),
                        'memory_entries': len(attempt['memory_entries']),
                        'thinking_entries': len(attempt['thinking_entries']),
                        'evaluation_entries': len(attempt['evaluation_entries']),
                        'urls_visited': len(attempt['urls_visited']),
                        'evaluator_feedback': attempt['evaluator_feedback']
                    }
                    for attempt in previous_attempts
                ]
                logger.info(f"Added retry metadata and previous attempts summary to result data")
            
            # Save to files
            await self._save_result(task.task_id, result_data)
            await self._save_agent_history(task.task_id, agent_history, retry_info.get('retry_attempt', 0) if retry_info else 0)
            logger.info(f"Saved task results to saved_trajectories/webarena/{task.task_id}.json")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error running task {task.task_id}: {str(e)}", exc_info=True)
            error_result = {
                'task_id': task.task_id,
                'success': False,
                'score': 0.0,
                'reasoning': f'Task execution error: {str(e)}',
                'functional_correctness': False,
                'error': str(e)
            }
            
            # Add retry information to error result if this is a retry
            if retry_info:
                error_result.update(retry_info)
            
            return error_result
        finally:
            # Remove from active sessions but don't stop the browser here
            # The browser will be stopped by cleanup_browser_safe in service.py
            if task.task_id in self.active_sessions:
                del self.active_sessions[task.task_id]
    
    async def _save_result(self, task_id: str, result_data: Dict):
        """Save task result to file"""
        results_dir = Path("saved_trajectories/webarena")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = results_dir / f"{task_id}.json"
        
        async with await anyio.open_file(result_file, 'w') as f:
            await f.write(json.dumps(result_data, indent=2, default=str))

    async def _save_agent_history(self, task_id: str, agent_history: AgentHistoryList, retry_attempt: int = 0):
        """Save agent history directly to file"""
        history_dir = Path("saved_trajectories/webarena/agent_history")
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Include retry attempt in filename to preserve all attempts
        if retry_attempt > 0:
            history_file = history_dir / f"{task_id}_history_retry_{retry_attempt}.json"
        else:
            history_file = history_dir / f"{task_id}_history.json"
        
        # Convert history to serializable format
        history_data = {
            'history': [
                {
                    'model_output': h.model_output.model_dump() if h.model_output else None,
                    'result': [r.model_dump() for r in h.result] if h.result else None,
                    'state': h.state.to_dict() if h.state else None,
                    'metadata': h.metadata.model_dump() if h.metadata else None
                }
                for h in agent_history.history
            ]
        }
        
        async with await anyio.open_file(history_file, 'w') as f:
            await f.write(json.dumps(history_data, indent=2, default=str))
        logger.info(f"Saved agent history to {history_file}")

    async def _save_retry_statistics(self, results: List[Dict], timestamp: str):
        """Save retry statistics to a separate file for analysis"""
        retry_stats_dir = Path("saved_trajectories/webarena/retry_stats")
        retry_stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract retry information
        retry_data = []
        for result in results:
            if result.get('retry_attempt', 0) > 0:
                retry_data.append({
                    'task_id': result.get('task_id'),
                    'retry_attempt': result.get('retry_attempt'),
                    'total_attempts': result.get('total_attempts'),
                    'success': result.get('success'),
                    'original_failure_reason': result.get('original_failure_reason'),
                    'final_reasoning': result.get('reasoning'),
                    'score': result.get('score')
                })
        
        if retry_data:
            retry_stats_file = retry_stats_dir / f"retry_stats_{timestamp}.json"
            async with await anyio.open_file(retry_stats_file, 'w') as f:
                await f.write(json.dumps(retry_data, indent=2, default=str))
            logger.info(f"Saved retry statistics to {retry_stats_file}")
        
        return retry_data


async def run_webarena_evaluation(
    tasks: List[WebArenaTask],
    llm: BaseChatModel,
    eval_model: BaseChatModel,
    max_parallel: int = 3,
    headless: bool = True,
    max_steps: int = 20,
    max_retries: int = 0,
    retry_delay: int = 5
) -> Dict:
    """Run WebArena evaluation on multiple tasks with optional retry for failed tasks"""
    
    evaluator = WebArenaEvaluator(eval_model)
    task_runner = WebArenaTaskRunner(llm, evaluator, headless, max_steps)
    
    # Create semaphore for parallel execution
    semaphore = asyncio.Semaphore(max_parallel)
    
    async def run_with_semaphore(task):
        async with semaphore:
            return await task_runner.run_task(task)
    
    async def run_with_semaphore_and_retry_info(task, retry_info=None):
        async with semaphore:
            return await task_runner.run_task(task, retry_info)
    
    # Run tasks in parallel
    logger.info(f"Running {len(tasks)} WebArena tasks with {max_parallel} parallel workers")
    logger.info(f"Max retries: {max_retries}, Retry delay: {retry_delay}s")
    
    # Track all results including retries
    all_results = []
    failed_tasks = []
    
    # Initial run
    initial_results = await asyncio.gather(*[run_with_semaphore(task) for task in tasks])
    
    # Process initial results and identify failed tasks
    for i, result in enumerate(initial_results):
        all_results.append(result)
        if not result.get('success', False):
            failed_tasks.append((tasks[i], result))
    
    # Retry failed tasks
    for retry_attempt in range(1, max_retries + 1):
        if not failed_tasks:
            logger.info(f"No failed tasks to retry after attempt {retry_attempt - 1}")
            break
            
        logger.info(f"Retry attempt {retry_attempt}/{max_retries}: Retrying {len(failed_tasks)} failed task(s)")
        logger.info(f"Failed tasks to retry: {[task.task_id for task, _ in failed_tasks]}")
        
        # Wait before retry
        if retry_delay > 0:
            logger.info(f"Waiting {retry_delay} seconds before retry...")
            await asyncio.sleep(retry_delay)
        
        # Retry failed tasks with retry information
        retry_tasks = [task for task, _ in failed_tasks]
        retry_info_list = [
            {
                'retry_attempt': retry_attempt,
                'total_attempts': retry_attempt + 1,
                'original_failure_reason': original_result.get('reasoning', 'Unknown')
            }
            for _, original_result in failed_tasks
        ]
        
        logger.info(f"Preparing retry info for {len(retry_tasks)} tasks:")
        for i, (task, original_result) in enumerate(failed_tasks):
            logger.info(f"  • Task {task.task_id}: retry_attempt={retry_attempt}, original_reason={original_result.get('reasoning', 'Unknown')[:100]}...")
        
        retry_results = await asyncio.gather(*[
            run_with_semaphore_and_retry_info(task, retry_info) 
            for task, retry_info in zip(retry_tasks, retry_info_list)
        ])
        
        # Update results and identify still-failed tasks
        new_failed_tasks = []
        for i, (original_task, original_result) in enumerate(failed_tasks):
            retry_result = retry_results[i]
            
            logger.info(f"Retry result for task {original_task.task_id}: success={retry_result.get('success', False)}")
            
            # Update the result in all_results if retry was successful
            if retry_result.get('success', False):
                # Find and update the original result
                for j, result in enumerate(all_results):
                    if result.get('task_id') == original_task.task_id:
                        all_results[j] = retry_result
                        logger.info(f"Task {original_task.task_id} succeeded on retry attempt {retry_attempt}")
                        break
            else:
                # Task still failed, update the result in all_results with retry information
                for j, result in enumerate(all_results):
                    if result.get('task_id') == original_task.task_id:
                        all_results[j] = retry_result
                        break
                new_failed_tasks.append((original_task, retry_result))
                logger.info(f"Task {original_task.task_id} failed on retry attempt {retry_attempt}")
                logger.debug(f"Retry failure reason: {retry_result.get('reasoning', 'Unknown')[:200]}...")
        
        failed_tasks = new_failed_tasks
        logger.info(f"After retry attempt {retry_attempt}: {len(failed_tasks)} tasks still failed")
    
    # Calculate summary statistics
    total_tasks = len(all_results)
    successful_tasks = sum(1 for r in all_results if r.get('success', False))
    average_score = sum(r.get('score', 0.0) for r in all_results) / total_tasks if total_tasks > 0 else 0.0
    
    # Count retries - count all results that have retry_attempt field (indicating they were retried)
    total_retries = sum(1 for r in all_results if 'retry_attempt' in r and r.get('retry_attempt', 0) > 0)
    successful_retries = sum(1 for r in all_results if r.get('success', False) and 'retry_attempt' in r and r.get('retry_attempt', 0) > 0)
    
    # Also count total retry attempts across all tasks (for debugging)
    total_retry_attempts = sum(r.get('retry_attempt', 0) for r in all_results if 'retry_attempt' in r)
    
    summary = {
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'failed_tasks': total_tasks - successful_tasks,
        'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
        'average_score': average_score,
        'total_retries': total_retries,
        'total_retry_attempts': total_retry_attempts,
        'successful_retries': successful_retries,
        'retry_success_rate': successful_retries / total_retries if total_retries > 0 else 0.0,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"WebArena evaluation completed: {successful_tasks}/{total_tasks} successful ({summary['success_rate']:.2%})")
    if total_retries > 0:
        logger.info(f"Retry statistics: {successful_retries}/{total_retries} retries successful ({summary['retry_success_rate']:.2%})")
        logger.info(f"Total retry attempts across all tasks: {total_retry_attempts}")
    else:
        logger.info("No retries were performed")
    
    # Debug: Print retry information for each result
    for result in all_results:
        if 'retry_attempt' in result:
            logger.info(f"Task {result.get('task_id')}: retry_attempt={result.get('retry_attempt')}, success={result.get('success')}")
    
    return {
        'summary': summary,
        'results': all_results
    }


def get_llm(model_name: str) -> BaseChatModel:
    """Get LLM instance for given model name"""
    
    if model_name.startswith('gpt'):
        from browser_use.llm.openai.chat import ChatOpenAI
        model_config = SUPPORTED_MODELS[model_name]
        model_name = model_config['model_name']
        if model_name.startswith('gpt'):
            return ChatOpenAI(model=model_name, temperature=0.1)
        elif model_name.startswith('o'):
            # o-series models only support temperature=1 (default)
            return ChatOpenAI(model=model_name, temperature=1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    elif model_name.startswith('claude'):
        from browser_use.llm.anthropic.chat import ChatAnthropic
        model_map = {
            'claude-3-sonnet': 'claude-3-sonnet-20240229',
            'claude-3-haiku': 'claude-3-haiku-20240307'
        }
        return ChatAnthropic(model=model_map.get(model_name, model_name), temperature=0.1)
    
    elif model_name.startswith('gemini'):
        from browser_use.llm.google.chat import ChatGoogle
        return ChatGoogle(model="gemini-pro", temperature=0.1)
    
    elif model_name.startswith('qwen'):
        from browser_use.llm.openai.chat import ChatOpenAI
        model_config = SUPPORTED_MODELS[model_name]
        return ChatOpenAI(
            model=model_config['model_name'],
            base_url=model_config['base_url'],
            api_key=os.getenv(model_config['api_key_env']),
            temperature=0.1
        )
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run WebArena evaluation with browser-use')
    parser.add_argument('--model', type=str, default='gpt-4o', 
                        choices=list(SUPPORTED_MODELS.keys()),
                        help='Model to use for the agent')
    parser.add_argument('--eval-model', type=str, default='gpt-4o',
                        choices=list(SUPPORTED_MODELS.keys()),
                        help='Model to use for evaluation')
    parser.add_argument('--max-parallel', type=int, default=3,
                        help='Maximum number of parallel tasks')
    parser.add_argument('--max-steps', type=int, default=30,
                        help='Maximum steps per task')
    parser.add_argument('--headless', action='store_true',
                        help='Run browser in headless mode')
    parser.add_argument('--config-dir', type=str, default='config_files/webarena',
                        help='Directory containing WebArena task configs')
    parser.add_argument('--start', type=int, default=0,
                        help='Start task index')
    parser.add_argument('--end', type=int, default=None,
                        help='End task index (exclusive)')
    parser.add_argument('--max-retries', type=int, default=0,
                        help='Maximum number of retries for failed tasks')
    parser.add_argument('--retry-delay', type=int, default=5,
                        help='Retry delay in seconds between attempts')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    async def main():
        # Get LLM instances
        llm = get_llm(args.model)
        eval_model = get_llm(args.eval_model)
        
        # Load WebArena tasks
        all_tasks = load_webarena_tasks(args.config_dir)
        
        # Select task subset
        start_idx = args.start
        end_idx = args.end if args.end is not None else len(all_tasks)
        tasks = all_tasks[start_idx:end_idx]
        
        logger.info(f"Running WebArena evaluation on {len(tasks)} tasks")
        logger.info(f"Using agent model: {args.model}, eval model: {args.eval_model}")
        
        # Run evaluation
        results = await run_webarena_evaluation(
            tasks=tasks,
            llm=llm, 
            eval_model=eval_model,
            max_parallel=args.max_parallel,
            headless=args.headless,
            max_steps=args.max_steps,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'saved_trajectories/webarena_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save retry statistics if any retries were performed
        task_runner = WebArenaTaskRunner(llm, WebArenaEvaluator(eval_model), args.headless, args.max_steps)
        retry_stats = await task_runner._save_retry_statistics(results['results'], timestamp)
        
        # Print summary
        summary = results['summary']
        print(f"\n=== WebArena Evaluation Results ===")
        print(f"Total tasks: {summary['total_tasks']}")
        print(f"Successful: {summary['successful_tasks']}")
        print(f"Failed: {summary['failed_tasks']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Average score: {summary['average_score']:.3f}")

        # Show retry statistics if any retries were performed
        if summary['total_retries'] > 0:
            print(f"\n=== Retry Statistics ===")
            # print(f"Total retries: {summary['total_retries']}")
            print(f"Total retry attempts: {summary['total_retry_attempts']}")
            print(f"Successful retries: {summary['successful_retries']}")
            print(f"Retry success rate: {summary['retry_success_rate']:.2%}")
            print(f"Retry statistics saved to: saved_trajectories/webarena/retry_stats/retry_stats_{timestamp}.json")
        
        print(f"Results saved to: {results_file}")
    
    # Run the evaluation
    asyncio.run(main())