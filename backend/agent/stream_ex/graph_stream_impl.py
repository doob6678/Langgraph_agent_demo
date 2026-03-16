import json
import time
import uuid
from typing import Any, Dict, Optional

from backend.agent.graph_async import async_agent_graph
from backend.agent.state_ex.agent_state import AgentState
from backend.agent.stream_ex.image_buffer import ImageMarkdownBuffer

def _openai_chunk(chat_id: str, created: int, model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }

def _sse_data(obj: Any) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"

async def stream_chat_graph(state: AgentState, chat_id: str, created: int):
    model = "doubao-seed-2-0-lite-260215" # Default model
    
    # Send initial role
    yield _sse_data(_openai_chunk(chat_id, created, model, {"role": "assistant"}))

    final_state = state
    image_buffer = ImageMarkdownBuffer()
    
    # Use astream_events to capture detailed events
    async for event in async_agent_graph.astream_events(state, version="v1"):
        kind = event["event"]
        
        # Capture LLM streaming tokens
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                safe_content = image_buffer.process(content)
                if safe_content:
                    yield _sse_data(_openai_chunk(chat_id, created, model, {"content": safe_content}))
        
        # Capture Tool execution
        elif kind == "on_tool_start":
            tool_name = event["name"]
            tool_input = event["data"].get("input")
            # We can send a tool call event if needed, but for now frontend focuses on content
            # If we want to show tool calls in frontend, we can emit a custom event
            yield _sse_data({
                "id": chat_id,
                "object": "chat.completion.chunk", 
                "created": created,
                "model": model,
                "choices": [],
                "x_tool_event": {
                    "tool": tool_name,
                    "status": "started",
                    "input": tool_input
                }
            })
            
        elif kind == "on_tool_end":
            tool_name = event["name"]
            tool_output = event["data"].get("output")
            # Convert tool output to string preview
            output_str = str(tool_output)
            # if tool output is very long (like base64 image or long text), truncate it for preview
            # but for rag_image_search, we want to keep the filenames visible
            if len(output_str) > 2000:
                output_str = output_str[:2000] + "..."
                
            yield _sse_data({
                "id": chat_id,
                "object": "chat.completion.chunk", 
                "created": created,
                "model": model,
                "choices": [],
                "x_tool_event": {
                    "tool": tool_name,
                    "status": "completed",
                    "result_preview": output_str
                }
            })
            
        # Update final state from chain end
        elif kind == "on_chain_end":
            # The top-level chain end event contains the final state
            if event["name"] == "LangGraph":
                output = event["data"].get("output")
                if output:
                    if isinstance(output, dict):
                        # If it's a dict, update final_state's fields
                        for k, v in output.items():
                            if hasattr(final_state, k):
                                setattr(final_state, k, v)
                    elif isinstance(output, AgentState):
                        final_state = output

    # Flush remaining buffer
    remaining = image_buffer.flush()
    if remaining:
        yield _sse_data(_openai_chunk(chat_id, created, model, {"content": remaining}))

    # After graph execution finishes
    # We need to construct the final response metadata
    
    # Send final stop
    yield _sse_data(_openai_chunk(chat_id, created, model, {}, finish_reason="stop"))
    
    x_final = {
        "response": getattr(final_state, "answer", "Generated via LangGraph Stream"),
        "images": getattr(final_state, "images", []),
        "search_results": getattr(final_state, "search_results", []),
        "timing": {},
        "metadata": getattr(final_state, "metadata", {}) or {}
    }
    
    yield _sse_data({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {"total_tokens": 0}, # Placeholder
        "x_final": x_final
    })
    
    yield _sse_done()
