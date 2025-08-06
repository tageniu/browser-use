"""GAIA 2023 dataset with Browser-Use integration.

Setup Instructions:
1. Create a Google Drive folder for all GAIA attachments
2. Upload all GAIA attachment files to this folder
3. Right-click the folder and select "Get link"
4. Set sharing to "Anyone with the link can view"
5. Copy the folder URL and set it in GAIA_ATTACHMENTS_FOLDER_URL below
6. The agent will navigate to this folder and click on specific files as needed
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any
import re
import random
from collections import Counter
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment
import tempfile
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

# Google Drive folder containing all GAIA attachments
# Replace with your actual Google Drive folder URL
# Make sure the folder is shared with "Anyone with the link can view"
GAIA_ATTACHMENTS_FOLDER_URL = "https://drive.google.com/drive/folders/1vnbNe_bs88VCHMG3ZrzW72WzXQ0__BK4?usp=sharing"
# Example: "https://drive.google.com/drive/folders/1vnbNe_bs88VCHMG3ZrzW72WzXQ0__BK4?usp=sharing"


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


def convert_audio_to_text(audio_file_path: str) -> str:
    """Convert audio file to text using speech recognition."""
    recognizer = sr.Recognizer()
    
    try:
        # Check if file exists
        if not os.path.exists(audio_file_path):
            return f"[Audio transcription failed: File not found at {audio_file_path}]"
        
        file_size = os.path.getsize(audio_file_path)
        logger.info(f"Processing audio file: {audio_file_path} (size: {file_size} bytes)")
        
        # Check file extension
        file_ext = Path(audio_file_path).suffix.lower()
        
        # If not WAV, convert to WAV first using pydub
        if file_ext != '.wav':
            logger.info(f"Converting {file_ext} to WAV format...")
            
            try:
                # Load audio file with pydub
                audio = AudioSegment.from_file(audio_file_path)
                
                # Convert to mono and standard sample rate for speech recognition
                audio = audio.set_channels(1)
                audio = audio.set_frame_rate(16000)
                
                # Add a small silence at the beginning to prevent cutting off start
                silence = AudioSegment.silent(duration=200)  # 200ms silence
                audio = silence + audio
                
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                    tmp_wav_path = tmp_wav.name
                    # Export as WAV
                    audio.export(tmp_wav_path, format='wav')
                
                # Use the temporary WAV file
                audio_file_path = tmp_wav_path
                cleanup_temp = True
            except Exception as e:
                logger.error(f"Failed to convert audio: {e}")
                return f"[Audio transcription failed: Could not convert audio - {e}. Make sure ffmpeg is installed]"
        else:
            cleanup_temp = False
        
        # Now process the WAV file
        try:
            with sr.AudioFile(audio_file_path) as source:
                # Don't adjust for ambient noise - it can cut off the beginning
                # Record the entire audio
                audio_data = recognizer.record(source)
        finally:
            # Clean up temporary file if created
            if cleanup_temp and os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            
        # Try Google Speech Recognition first (free, no API key needed)
        try:
            # Get the transcription
            result = recognizer.recognize_google(audio_data, show_all=True)
            
            if isinstance(result, dict) and 'alternative' in result:
                # Get the best transcription
                text = result['alternative'][0]['transcript']
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)
                
            logger.info(f"Successfully transcribed audio: {text[:100]}...")
            return text
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand the audio")
            return "[Audio transcription failed: Could not understand the audio]"
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition error: {e}")
            return f"[Audio transcription failed: {e}]"
            
    except Exception as e:
        logger.error(f"Failed to process audio file: {e}")
        import traceback
        traceback.print_exc()
        return f"[Audio transcription failed: {e}]"


async def run_gaia_task_with_agent(task_data: Dict[str, Any], task_index: int, attempt_number: int = 1) -> Dict[str, Any]:
    """Run a single GAIA task attempt using Browser-Use agent."""
    
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
    
    # Check for attachments
    file_name = task_data.get("file_name", "")
    file_path = task_data.get("file_path", "")
    
    # GAIA system prompt for answer format
    system_prompt = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. 
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""
    
    # Handle attachments if present
    attachment_info = ""
    if file_name:
        # Check if it's an audio file
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.opus'}
        file_ext = Path(file_name).suffix.lower()
        
        if file_ext in audio_extensions:
            logger.info(f"Detected audio file: {file_name}")
            
            # If we have the local file path, convert it
            if file_path and os.path.exists(file_path):
                logger.info(f"Converting audio file to text: {file_path}")
                audio_text = convert_audio_to_text(file_path)
                attachment_info = f"\n\nATTACHMENT: {file_name} (Audio file)\n\nTranscribed content:\n{audio_text}"
            else:
                # Fallback to Google Drive approach but with a note about audio
                attachment_info = f"\n\nATTACHMENT: {file_name} (Audio file)\n\nNote: This is an audio file. The content needs to be transcribed to answer the question."
        else:
            attachment_info = f"""\n\nATTACHMENT: {file_name}

To analyze this attachment:
1. Navigate to the Google Drive folder: {GAIA_ATTACHMENTS_FOLDER_URL}
2. Wait for the folder to load completely
3. Look for '{file_name}' in the file list (you may need to scroll)
4. Click on '{file_name}' to open it
5. The file will open in Google's appropriate viewer:
   - Excel files (.xlsx) → Google Sheets
   - PowerPoint files (.pptx) → Google Slides
   - PDFs → Google's PDF viewer
   - Images → Direct image view
6. IMPORTANT: For multi-page documents (PDFs, presentations, etc.):
   - Use SCROLL DOWN to view subsequent pages within the same document
   - DO NOT click the "Next" button - it navigates to a different file
   - Keep scrolling until you've reviewed all pages of the current document
7. Analyze the content to answer the question
8. Use the browser's vision capabilities for images and visual content"""
            logger.info(f"Task has attachment: {file_name} in Google Drive folder")
    
    # Create the task prompt with GAIA formatting requirements
    task_prompt = f"""{system_prompt}

Question: {task_question}{attachment_info}

IMPORTANT: Remember to conclude your response with "FINAL ANSWER: [YOUR FINAL ANSWER]" following the format rules above.
    """
    
    logger.info(f"Running GAIA Task {task_index + 1} - Attempt {attempt_number}")
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Level: {task_level}")
    logger.info(f"Question: {task_question}")
    if file_name:
        logger.info(f"Attachment: {file_name}")
    
    # Log the expected/true answer
    expected_answer = task_data.get("Final answer", "")
    if expected_answer:
        logger.info(f"Expected Answer: {expected_answer}")
    else:
        logger.info("Expected Answer: Not available")
    
    # Create agent
    browser_config = BrowserConfig(
		headless=False,
		window_size={'width': 1720, 'height': 1080},
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
        
        logger.info(f"✓ Task {task_index + 1} Attempt {attempt_number} completed in {execution_time:.2f} seconds")
        
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
        logger.error(f"✗ Task {task_index + 1} Attempt {attempt_number} failed: {e}")
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


async def run_task_with_multiple_attempts(task_data: Dict[str, Any], task_index: int, num_attempts: int = 3) -> Dict[str, Any]:
    """Run a GAIA task multiple times and use majority voting for the final answer."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running task {task_index + 1} with {num_attempts} attempts")
    logger.info(f"{'='*60}")
    
    # Run the task multiple times
    all_attempts = []
    for attempt_num in range(1, num_attempts + 1):
        logger.info(f"\n--- Attempt {attempt_num}/{num_attempts} ---")
        attempt_result = await run_gaia_task_with_agent(task_data, task_index, attempt_num)
        all_attempts.append(attempt_result)
        
        # Small delay between attempts
        if attempt_num < num_attempts:
            await asyncio.sleep(2)
    
    # Extract answers from all attempts
    answers = []
    for i, attempt in enumerate(all_attempts):
        answer = attempt.get("agent_answer", "No answer extracted")
        answers.append(answer)
        logger.info(f"Attempt {i+1} answer: {answer}")
    
    # Perform majority voting
    answer_counts = Counter(answers)
    
    # Get the most common answer(s)
    if answer_counts:
        max_count = max(answer_counts.values())
        most_common_answers = [ans for ans, count in answer_counts.items() if count == max_count]
        
        if len(most_common_answers) == 1:
            # Clear majority
            final_answer = most_common_answers[0]
            logger.info(f"\nMajority vote result: {final_answer} (appeared {max_count} times)")
        else:
            # Tie - randomly pick one
            final_answer = random.choice(most_common_answers)
            logger.info(f"\nTie detected among {len(most_common_answers)} answers: {most_common_answers}")
            logger.info(f"Randomly selected: {final_answer}")
    else:
        # No valid answers
        final_answer = "No answer extracted"
        logger.info("\nNo valid answers from any attempt")
    
    # Build the final result
    result = {
        "task_index": task_index,
        "task_id": task_data.get("task_id", f"task_{task_index}"),
        "question": task_data.get("Question", ""),
        "level": task_data.get("Level", ""),
        "expected_answer": task_data.get("Final answer", ""),
        "final_answer": final_answer,
        "all_attempts": all_attempts,
        "attempt_answers": answers,
        "answer_distribution": dict(answer_counts),
        "num_attempts": num_attempts,
        "complete": any(attempt.get("complete", False) for attempt in all_attempts),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Score the final answer if expected answer is available
    expected_answer = task_data.get("Final answer", "")
    if expected_answer and expected_answer != "?":
        try:
            logger.info(f"Expected Answer: {expected_answer}")
            is_correct = question_scorer(final_answer, expected_answer)
            logger.info(f"Final Answer Correct: {is_correct}")
            result["success"] = is_correct
        except Exception as e:
            logger.warning(f"Could not score answer: {e}")
    
    return result


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
    
    # Initialize file names
    results_file = "gaia_browser_use_results.json"
    answers_file = "gaia_answers.json"
    
    # Run tasks sequentially to avoid resource conflicts
    for i, task_data in enumerate(tasks_to_run):
        # if i == 4 or i == 26 or i == 33: # For youtube tasks
        if i == 6: # For PDF task
        # if i == 7: # For attachment task in [7, 9, 27, 30, 34, 44, 51]
            logger.info("\n")
            logger.info(f"{'-'*60}")
            logger.info(f"TASK {i + 1}/{len(tasks_to_run)}")
            logger.info(f"{'-'*60}")
            
            # Convert task_data to dict if needed
            if not isinstance(task_data, dict):
                task_data = dict(task_data)
            
            # Run task with multiple attempts and majority voting
            result = await run_task_with_multiple_attempts(task_data, i, num_attempts=3)
            all_results.append(result)
            
            # Save individual result to files after each task
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            
            # Save answer in JSONL format after each task
            answer_obj = {
                "task_id": result.get("task_id", f"task_{i}"),
                "model_answer": result.get("final_answer", "No answer extracted")
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
    accuracy_rate = len(correct_tasks) / len(all_results) * 100 if all_results else 0
    
    logger.info(f"Tasks completed: {len(all_results)}")
    logger.info(f"Tasks with at least one successful attempt: {len(successful_tasks)}")
    logger.info(f"Tasks answered correctly (based on majority vote): {len(correct_tasks)}")
    logger.info(f"Score: {accuracy_rate:.1f}%")
    
    # Calculate total execution time from all attempts
    total_time = 0
    total_attempts = 0
    for result in all_results:
        if "all_attempts" in result:
            for attempt in result["all_attempts"]:
                total_time += attempt.get("execution_time_seconds", 0)
                total_attempts += 1
    
    avg_time_per_attempt = total_time / total_attempts if total_attempts else 0
    avg_time_per_task = total_time / len(all_results) if all_results else 0
    
    logger.info(f"\nTotal execution time: {total_time:.1f} seconds")
    logger.info(f"Average time per task (3 attempts): {avg_time_per_task:.1f} seconds") 
    logger.info(f"Average time per attempt: {avg_time_per_attempt:.1f} seconds")
    
    # Print voting statistics
    logger.info("\nVoting Statistics:")
    unanimous_tasks = 0
    majority_tasks = 0
    tie_tasks = 0
    
    for result in all_results:
        if "answer_distribution" in result:
            distribution = result["answer_distribution"]
            if distribution:
                max_count = max(distribution.values())
                if max_count == 3:
                    unanimous_tasks += 1
                elif max_count == 2:
                    majority_tasks += 1
                else:
                    tie_tasks += 1
    
    logger.info(f"  Unanimous agreement (3/3): {unanimous_tasks} tasks")
    logger.info(f"  Majority agreement (2/3): {majority_tasks} tasks")
    logger.info(f"  No majority (tie): {tie_tasks} tasks")
    
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