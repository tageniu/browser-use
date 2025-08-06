# Video Analysis with Browser-Use

Browser-Use now includes powerful video analysis capabilities that allow AI agents to understand and interact with video content on web pages.

## Features

### 1. Video Frame Analysis
Extract frames from videos and analyze them using AI vision models to answer questions about video content.

```python
# The agent will automatically use video analysis when encountering video-related tasks
agent = Agent(
    task="What objects appear in this video? https://example.com/video",
    llm=llm,
    browser=browser,
)
```

### 2. Video Transcript Extraction
Extract captions, subtitles, or auto-generated transcripts from videos.

```python
agent = Agent(
    task="Extract and summarize the transcript from this YouTube video",
    llm=llm,
    browser=browser,
)
```

### 3. Video Snapshot Capture
Take screenshots of specific moments in videos for analysis or documentation.

```python
agent = Agent(
    task="Capture the key moments from this tutorial video",
    llm=llm,
    browser=browser,
)
```

## Available Actions

### `analyze_video`
Analyzes video content by extracting frames and using AI to answer questions.

Parameters:
- `video_selector`: CSS selector or XPath of the video element (optional)
- `video_index`: Index of the video element if multiple exist (default: 0)
- `analysis_prompt`: What to analyze in the video
- `max_frames`: Maximum number of frames to extract (default: 10)
- `sample_interval`: Interval between frames in seconds (default: 1.0)

### `extract_video_transcript`
Extracts transcript or captions from a video.

Parameters:
- `video_selector`: CSS selector or XPath of the video element (optional)
- `video_index`: Index of the video element if multiple exist (default: 0)
- `language`: Language code for transcript (default: 'en')

### `take_video_snapshot`
Takes a snapshot of the current or specified video frame.

Parameters:
- `video_selector`: CSS selector or XPath of the video element (optional)
- `video_index`: Index of the video element if multiple exist (default: 0)
- `timestamp`: Timestamp in seconds to capture (optional, uses current time if not specified)

## How It Works

1. **Frame Extraction**: The system uses OpenCV to extract frames from videos at specified intervals
2. **AI Analysis**: Extracted frames are sent to the vision-capable LLM for analysis
3. **Temporal Understanding**: Multiple frames allow the AI to understand changes over time
4. **Smart Sampling**: Frames are sampled intelligently to capture key moments while avoiding redundancy

## Best Practices

1. **Use Appropriate Frame Counts**: 
   - For quick identification tasks: 3-5 frames
   - For counting or tracking objects: 10-20 frames
   - For detailed analysis: Adjust based on video length

2. **Optimize Sample Intervals**:
   - Fast-paced content: 0.5-1 second intervals
   - Slow content: 2-5 second intervals
   - Long videos: Increase interval to avoid too many frames

3. **Leverage Transcripts First**:
   - For YouTube videos, try extracting transcripts first
   - Transcripts are faster and often contain the information you need
   - Use frame analysis when visual information is crucial

4. **Combine Methods**:
   - Use transcripts for context
   - Use frame analysis for visual details
   - Use snapshots for documentation

## Example Use Cases

### Counting Objects in Videos
```python
task = "Count the maximum number of birds visible at once in this nature video"
```

### Tutorial Analysis
```python
task = "Analyze this cooking tutorial and list all ingredients shown"
```

### Video Summarization
```python
task = "Watch this presentation and summarize the key points with visual examples"
```

### Quality Checking
```python
task = "Check if this video contains any inappropriate content"
```

## Performance Considerations

- Video analysis requires downloading video content, which may take time for large files
- Frame extraction is optimized but still requires processing time
- Consider using lower frame counts for faster results
- The system automatically resizes frames to optimize for LLM processing

## Limitations

- Videos must be accessible via standard HTML5 video elements
- Some streaming platforms may have DRM or other protections
- Very long videos may require sampling strategies to stay within context limits
- Frame quality depends on the original video quality