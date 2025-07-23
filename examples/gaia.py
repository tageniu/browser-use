"""GAIA 2023 dataset with Browser-Use integration."""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any
import re

import datasets
from dotenv import load_dotenv

# Import browser-use components
from browser_use import Agent
from browser_use.browser import Browser, BrowserConfig
from browser_use.llm.openai.chat import ChatOpenAI

# Import GAIA scorer
from gaia_scorer import question_scorer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CITATION = """ """
_DESCRIPTION = """ """
_HOMEPAGE = ""
_LICENSE = ""
_NAMES = [
    "2023_all",
    "2023_level1",
    "2023_level2",
    "2023_level3",
]

YEAR_TO_LEVELS = {"2023": [1, 2, 3]}

separator = "_"


class GAIA_dataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=name, version=version, description=name)
        for name, version in zip(_NAMES, [VERSION] * len(_NAMES))
    ]

    def _info(self):
        features = datasets.Features(
            {
                "task_id": datasets.Value("string"),
                "Question": datasets.Value("string"),
                "Level": datasets.Value("string"),
                "Final answer": datasets.Value("string"), # ? for test values
                "file_name": datasets.Value("string"),
                "file_path": datasets.Value("string"),  # generated here
                "Annotator Metadata": {k: datasets.Value("string") for k in ["Steps", "Number of steps", "How long did this take?", "Tools", "Number of tools"]} # "", 
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        year, level_name = self.config.name.split(separator)
        if level_name == "all":
            levels = YEAR_TO_LEVELS[year]
        else:
            level_name = int(level_name.split("level")[1])
            levels = [level_name]
        print(year, level_name)

        output = []
        for split in ["test", "validation"]:
            root_file = dl_manager.download(os.path.join(year, split, "metadata.jsonl"))
            test_attached_files = {"": ""}
            with open(root_file, mode="r", encoding="utf-8") as f:
                for line in f:
                    cur_line = json.loads(line)
                    if cur_line["Level"] in levels and cur_line["file_name"] != "":
                        attached_file_name = cur_line["file_name"]
                        attached_file = dl_manager.download(os.path.join(year, split, attached_file_name))
                        test_attached_files[attached_file_name] = attached_file

            output.append(
                datasets.SplitGenerator(
                    name=getattr(datasets.Split, split.upper()),
                    gen_kwargs={"root_file": root_file, "attached_files": test_attached_files, "levels": levels},
                )
            )
        return output

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, root_file: str, attached_files: dict, levels: list[int]):
        with open(root_file, "r", encoding="utf-8") as f:
            for key, line in enumerate(f):
                cur_line = json.loads(line)
                if cur_line["Level"] in levels:
                    cur_line["file_path"] = attached_files[cur_line["file_name"]]
                    yield key, cur_line


async def run_gaia_task_with_agent(task_data: Dict[str, Any], task_index: int) -> Dict[str, Any]:
    """Run a single GAIA task using Browser-Use agent."""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.1,  # Lower temperature for accuracy
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Format the task for the agent
    task_question = task_data.get("Question", "")
    task_level = task_data.get("Level", "")
    task_id = task_data.get("task_id", f"task_{task_index}")
    
    # GAIA system prompt for answer format
    system_prompt = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. 
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""
    
    # Create the task prompt with GAIA formatting requirements
    task_prompt = f"""{system_prompt}

Question: {task_question}

IMPORTANT: Remember to conclude your response with "FINAL ANSWER: [YOUR FINAL ANSWER]" following the format rules above.
    """
    
    logger.info(f"Running GAIA Task {task_index + 1}")
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Level: {task_level}")
    logger.info(f"Question: {task_question}")
    
    # Log the expected/true answer
    expected_answer = task_data.get("Final answer", "")
    if expected_answer:
        logger.info(f"Expected Answer: {expected_answer}")
    else:
        logger.info("Expected Answer: Not available")
    
    # Create agent
    browser_config = BrowserConfig(
		headless=False,
		window_size={'width': 1820, 'height': 1080},
	)
    browser = Browser(browser_profile=browser_config)
    agent = Agent(
        task=task_prompt,
        llm=llm,
        browser=browser,
        use_vision=True,
        # save_conversation_path=f"gaia_task_{task_index + 1}_{task_id}.md",
    )
    
    start_time = datetime.now()
    
    try:
        history = await agent.run(max_steps=15)  # More steps for complex GAIA tasks
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✓ Task {task_index + 1} completed in {execution_time:.2f} seconds")
        
        # Extract the FINAL ANSWER from the agent's response
        final_result = "No answer extracted"
        
        # Try to extract the FINAL ANSWER from the agent's history
        try:
            # Check if the agent completed with a done action
            if history.is_done():
                # Get the last step's result
                last_step = history.history[-1]
                if last_step.result and last_step.result[-1].extracted_content:
                    # The done action's text is stored in extracted_content
                    done_text = last_step.result[-1].extracted_content
                    
                    # Search for FINAL ANSWER pattern in the done text
                    match = re.search(r'FINAL ANSWER:\s*(.+?)(?:\n|$)', done_text, re.IGNORECASE)
                    if match:
                        final_result = match.group(1).strip()
                        logger.info(f"Extracted FINAL ANSWER: {final_result}")
                    else:
                        # Try to find any mention of final answer in different formats
                        alt_match = re.search(r'(?:final answer is|my final answer|answer is):\s*(.+?)(?:\n|$)', done_text, re.IGNORECASE)
                        if alt_match:
                            final_result = alt_match.group(1).strip()
                            logger.info(f"Extracted answer (alternative format): {final_result}")
                        else:
                            # If no pattern found, the whole done text might be the answer
                            logger.info(f"No FINAL ANSWER pattern found, using done text: {done_text}")
                            final_result = done_text
            else:
                logger.warning("Agent did not complete with a done action")
        except Exception as e:
            logger.error(f"Failed to extract final answer from history: {e}")
            pass
        
        result = {
            "task_index": task_index,
            "task_id": task_id,
            "question": task_question,
            "level": task_level,
            "expected_answer": task_data.get("Final answer", ""),
            "agent_answer": final_result,
            "execution_time_seconds": execution_time,
            "complete": True,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Score the answer if expected answer is available
        if expected_answer and expected_answer != "?":
            try:
                logger.info(f"Expected Answer: {expected_answer}")
                is_correct = question_scorer(final_result, expected_answer)
                logger.info(f"Answer Correct: {is_correct}")
                result["success"] = is_correct
            except Exception as e:
                logger.warning(f"Could not score answer: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Task {task_index + 1} failed: {e}")
        return {
            "task_index": task_index,
            "task_id": task_id,
            "question": task_question,
            "level": task_level,
            "expected_answer": task_data.get("Final answer", ""),
            "agent_answer": None,
            "error": str(e),
            "complete": False,
            "timestamp": datetime.now().isoformat(),
        }


async def load_and_run_gaia_tasks(num_tasks: int = 10):
    """Load GAIA dataset and run the first N tasks with Browser-Use agent."""
    
    logger.info("="*80)
    logger.info("LOADING REAL GAIA DATASET AND RUNNING TASKS WITH BROWSER-USE")
    logger.info("="*80)
    
    try:
        # Load the real GAIA dataset
        logger.info("Loading GAIA dataset from Hugging Face...")
        logger.info("Note: This requires authentication for the gated dataset")
        
        dataset = datasets.load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
        dataset_list = list(dataset)
        
        logger.info(f"✓ GAIA dataset loaded successfully with {len(dataset_list)} tasks")
        
    except Exception as e:
        logger.error(f"Failed to load GAIA dataset: {e}")
        logger.error("The GAIA dataset is gated on Hugging Face. Please:")
        logger.error("1. Visit https://huggingface.co/datasets/gaia-benchmark/GAIA")
        logger.error("2. Request access to the dataset")
        logger.error("3. Authenticate with: huggingface-cli login")
        return []
    
    logger.info(f"Running first {num_tasks} tasks...")
    
    # Take only the first num_tasks
    tasks_to_run = dataset_list[:min(num_tasks, len(dataset_list))]
    
    all_results = []
    
    # Run tasks sequentially to avoid resource conflicts
    for i, task_data in enumerate(tasks_to_run):
        if i == 4 or i == 26 or i == 33: # For youtube tasks
            logger.info("\n")
            logger.info(f"{'-'*60}")
            logger.info(f"TASK {i + 1}/{len(tasks_to_run)}")
            logger.info(f"{'-'*60}")
            
            # Convert task_data to dict if needed
            if not isinstance(task_data, dict):
                task_data = dict(task_data)
            
            result = await run_gaia_task_with_agent(task_data, i)
            all_results.append(result)
            
            # Save individual result to files after each task
            results_file = "gaia_browser_use_results.json"
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            
            # Save answer in JSONL format after each task
            answers_file = "gaia_answers.json"
            answer_obj = {
                "task_id": result.get("task_id", f"task_{i}"),
                "model_answer": result.get("agent_answer", "No answer extracted")
            }
            
            # Append to answers file (or create if first task)
            mode = "a" if i > 0 else "w"
            with open(answers_file, mode) as f:
                f.write(json.dumps(answer_obj) + "\n")
            
            logger.info(f"Results saved to: {results_file}")
            logger.info(f"Answer saved to: {answers_file}")
            
            # Small delay between tasks
            await asyncio.sleep(1)

    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    successful_tasks = [r for r in all_results if r.get("complete", False)]
    correct_tasks = [r for r in all_results if r.get("success", False)]
    success_rate = len(successful_tasks) / len(all_results) * 100 if all_results else 0
    accuracy_rate = len(correct_tasks) / len(all_results) * 100 if all_results else 0
    
    logger.info(f"Tasks completed: {len(all_results)}")
    logger.info(f"Tasks ran successfully: {len(successful_tasks)}")
    logger.info(f"Tasks answered correctly: {len(correct_tasks)}")
    logger.info(f"Score: {accuracy_rate:.1f}%")
    
    total_time = sum(r.get("execution_time_seconds", 0) for r in all_results)
    avg_time = total_time / len(all_results) if all_results else 0
    logger.info(f"Total execution time: {total_time:.1f} seconds")
    logger.info(f"Average time per task: {avg_time:.1f} seconds")
    
    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info(f"Answers saved to: {answers_file}")
    logger.info("Note: Results and answers were saved incrementally after each task")

    return all_results


def main():
    num_tasks = 53  # Number of tasks to run
    
    logger.info("Browser-Use GAIA Dataset Runner")
    logger.info(f"This will load the real GAIA dataset and run the first {num_tasks} tasks")
    
    # Run the tasks
    asyncio.run(load_and_run_gaia_tasks(num_tasks=num_tasks))

if __name__ == "__main__":
    main()