# =========================================================================================================================
# source .venv/bin/activate
# Apple Trade-in
# python eval/webarena_service.py --model gpt-o4-mini --eval-model gpt-o4-mini --start 812 --end 813 --max-steps 15
# python eval/webarena_service.py --model gpt-4.1 --eval-model gpt-4.1 --start 812 --end 813 --max-steps 15
# =========================================================================================================================

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
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --max-parallel 10 --start 401 --end 411 --max-steps 15
# python eval/webarena_service.py --model qwen --eval-model gpt-4.1 --max-parallel 10 --start 401 --end 411 --max-steps 15
# python eval/webarena_service.py --model gpt-o4-mini --eval-model gpt-o4-mini --max-parallel 10 --start 401 --end 411 --max-steps 15
# Shopping
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --max-parallel 10 --start 141 --end 151 --max-steps 15
# python eval/webarena_service.py --model qwen --eval-model gpt-4.1 --max-parallel 10 --start 141 --end 151 --max-steps 15
# python eval/webarena_service.py --model gpt-o4-mini --eval-model gpt-o4-mini --max-parallel 10 --start 141 --end 151 --max-steps 15
# Shopping admin
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --max-parallel 10 --start 111 --end 121 --max-steps 15
# python eval/webarena_service.py --model qwen --eval-model gpt-4.1 --max-parallel 10 --start 111 --end 121 --max-steps 15
# Gitlab
# python eval/webarena_service.py --model gpt-4o --eval-model gpt-4o --max-parallel 10 --start 303 --end 313 --max-steps 15
# python eval/webarena_service.py --model qwen --eval-model gpt-4.1 --max-parallel 10 --start 303 --end 313 --max-steps 15
# =========================================================================================================================

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
    
    async def run_task(self, task: WebArenaTask) -> Dict:
        """Run a single WebArena task"""
        logger.info(f"Starting WebArena task: {task.task_id}")
        logger.info(f"Task intent: {task.intent}")
        logger.info(f"Start URL: {task.start_url}")
        logger.info(f"Target sites: {task.sites}")
        logger.info(f"Requires login: {task.require_login}")
        
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
            
            # Create agent with task context
            agent_task = f"Navigate to {task.start_url} and {task.intent}"
            logger.info(f"Initializing agent with task: {agent_task}")
            
            # Configure memory interval to prevent information loss
            memory_config = MemoryConfig(
                agent_id=f"webarena_agent_{task.task_id}",
                memory_interval=20,
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
            
            # Evaluate the result
            logger.info(f"Evaluating task {task.task_id}")
            evaluation = await self.evaluator.evaluate_task_completion(task, agent_history)
            
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
            
            # Save to files
            await self._save_result(task.task_id, result_data)
            await self._save_agent_history(task.task_id, agent_history)
            logger.info(f"Saved task results to saved_trajectories/webarena/{task.task_id}.json")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error running task {task.task_id}: {str(e)}", exc_info=True)
            return {
                'task_id': task.task_id,
                'success': False,
                'score': 0.0,
                'reasoning': f'Task execution error: {str(e)}',
                'functional_correctness': False,
                'error': str(e)
            }
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

    async def _save_agent_history(self, task_id: str, agent_history: AgentHistoryList):
        """Save agent history directly to file"""
        history_dir = Path("saved_trajectories/webarena/agent_history")
        history_dir.mkdir(parents=True, exist_ok=True)
        
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


async def run_webarena_evaluation(
    tasks: List[WebArenaTask],
    llm: BaseChatModel,
    eval_model: BaseChatModel,
    max_parallel: int = 3,
    headless: bool = True,
    max_steps: int = 20
) -> Dict:
    """Run WebArena evaluation on multiple tasks"""
    
    evaluator = WebArenaEvaluator(eval_model)
    task_runner = WebArenaTaskRunner(llm, evaluator, headless, max_steps)
    
    # Create semaphore for parallel execution
    semaphore = asyncio.Semaphore(max_parallel)
    
    async def run_with_semaphore(task):
        async with semaphore:
            return await task_runner.run_task(task)
    
    # Run tasks in parallel
    logger.info(f"Running {len(tasks)} WebArena tasks with {max_parallel} parallel workers")
    results = await asyncio.gather(*[run_with_semaphore(task) for task in tasks])
    
    # Calculate summary statistics
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get('success', False))
    average_score = sum(r.get('score', 0.0) for r in results) / total_tasks if total_tasks > 0 else 0.0
    
    summary = {
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'failed_tasks': total_tasks - successful_tasks,
        'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
        'average_score': average_score,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"WebArena evaluation completed: {successful_tasks}/{total_tasks} successful ({summary['success_rate']:.2%})")
    
    return {
        'summary': summary,
        'results': results
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
            max_steps=args.max_steps
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'saved_trajectories/webarena_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        summary = results['summary']
        print(f"\n=== WebArena Evaluation Results ===")
        print(f"Total tasks: {summary['total_tasks']}")
        print(f"Successful: {summary['successful_tasks']}")
        print(f"Failed: {summary['failed_tasks']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Average score: {summary['average_score']:.3f}")
        print(f"Results saved to: {results_file}")
    
    # Run the evaluation
    asyncio.run(main()) 