"""Example of using browser-use for video analysis tasks."""

import asyncio

from browser_use import Agent, Browser, Controller
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


async def analyze_youtube_video():
	"""Example: Analyze a YouTube video to count bird species."""
	
	# Initialize browser and agent
	browser = Browser()
	
	# You can use any vision-capable LLM model
	# Examples:
	llm = ChatAnthropic(model='claude-3-5-sonnet-20241022', temperature=0)
	# llm = ChatOpenAI(model='gpt-4o', temperature=0)
	
	agent = Agent(
		task="In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?",
		llm=llm,
		browser=browser,
	)
	
	# Let the agent run
	result = await agent.run()
	print(f"Agent result: {result}")


async def analyze_video_with_transcript():
	"""Example: Extract and analyze video transcript."""
	
	browser = Browser()
	llm = ChatAnthropic(model='claude-3-5-sonnet-20241022', temperature=0)
	
	agent = Agent(
		task="Go to https://www.youtube.com/watch?v=dQw4w9WgXcQ and extract the video transcript. Summarize the main themes.",
		llm=llm,
		browser=browser,
	)
	
	result = await agent.run()
	print(f"Transcript analysis: {result}")


async def custom_video_analysis():
	"""Example: Custom video analysis with specific parameters."""
	
	browser = Browser()
	llm = ChatAnthropic(model='claude-3-5-sonnet-20241022', temperature=0)
	
	# Create a custom controller with video analysis actions
	controller = Controller()
	
	# The agent will automatically use video analysis actions when needed
	agent = Agent(
		task="""Go to any video streaming site and find a nature documentary video. 
		Analyze the video to identify:
		1. Different types of animals shown
		2. The environments/habitats featured
		3. Any interesting behaviors observed
		
		Use the video analysis tools to extract frames and analyze the content.""",
		llm=llm,
		browser=browser,
		controller=controller,
	)
	
	result = await agent.run()
	print(f"Video analysis results: {result}")


async def video_frame_capture():
	"""Example: Capture specific moments from a video."""
	
	browser = Browser()
	llm = ChatAnthropic(model='claude-3-5-sonnet-20241022', temperature=0)
	
	agent = Agent(
		task="""Navigate to a video tutorial on YouTube about cooking. 
		Take snapshots at key moments:
		1. When ingredients are shown
		2. During important cooking steps
		3. The final result
		
		Save a summary of what you captured.""",
		llm=llm,
		browser=browser,
	)
	
	result = await agent.run()
	print(f"Captured moments: {result}")


if __name__ == "__main__":
	# Run the example you want to test
	asyncio.run(analyze_youtube_video())
	
	# Uncomment to run other examples:
	# asyncio.run(analyze_video_with_transcript())
	# asyncio.run(custom_video_analysis())
	# asyncio.run(video_frame_capture())