from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from backend.agent.config_ex.ark_config import configure_ark as _configure_ark
from backend.agent.node_ex.agent_node import agent_node
from backend.agent.state_ex.agent_state import AgentState
from backend.agent.stream_ex.stream_chat_with_tools import stream_chat_with_tools
from backend.agent.tool_ex.state_tool_runner import execute_tool_call_into_state as _execute_tool_call_into_state
from backend.agent.tool_ex.state_tool_runner import process_tool_results

load_dotenv()


def configure_ark(api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    return _configure_ark(api_key=api_key, base_url=base_url, model=model)


def create_agent_graph():
    """创建Agent图"""
    
    def should_continue(state: AgentState) -> str:
        """决定是否继续执行"""
        # 如果Agent标记需要工具调用，继续执行工具
        if state.needs_tool:
            return "continue"
        
        # 如果已有最终答案，结束
        if state.answer:
            return "end"
        
        # 否则继续调用工具
        return "continue"
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("process_results", process_tool_results)
    
    # 设置入口点
    workflow.set_entry_point("agent")
    
    # 添加条件边 - 直接到结果处理节点
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "process_results",
            "end": END
        }
    )
    
    # 添加结果处理到Agent的边
    workflow.add_edge("process_results", "agent")
    
    # 编译图
    return workflow.compile()

# 创建全局Agent图实例
agent_graph = create_agent_graph()
