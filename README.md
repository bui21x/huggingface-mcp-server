# HuggingFace MCP Server

A Model Context Protocol (MCP) server that provides HuggingFace model integration for AI agents.

## Features

- Model inference with GPU support
- Model caching for improved performance
- Multiple task support
- Custom parameter configuration
- Health monitoring with GPU status

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
HUGGINGFACE_TOKEN=your_token_here  # Optional
```

3. Run server:
```bash
uvicorn src.mcp_server:app --reload
```

## API Endpoints

- POST /inference - Run model inference
- GET /models - List loaded models
- GET /health - Check server health

## Example Usage

```python
# Run inference
POST /inference
{
    "model_id": "gpt2",
    "task": "text-generation",
    "inputs": "Once upon a time",
    "parameters": {
        "max_length": 50,
        "temperature": 0.7
    }
}
```

## MCP Integration

This server follows the MCP specification for tool integration with AI agents.