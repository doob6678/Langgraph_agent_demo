from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from backend.agent.config_ex.model_config import configure_model as _configure_model
from backend.agent.node_ex.agent_node_async import agent_node_async
from backend.agent.node_ex.memory_node import memory_recall_node, memory_store_node
from backend.agent.state_ex.agent_state import AgentState
from backend.agent.tool_ex.state_tool_runner import process_tool_results

load_dotenv()


def configure_model(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    return _configure_model(api_key=api_key, base_url=base_url, model=model, provider=provider)


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
    workflow.add_node("memory_recall", memory_recall_node)
    workflow.add_node("agent", agent_node_async)
    workflow.add_node("process_results", process_tool_results)
    workflow.add_node("memory_store", memory_store_node)
    
    # 设置入口点
    workflow.set_entry_point("memory_recall")
    
    # 记忆召回后直接进入 Agent
    workflow.add_edge("memory_recall", "agent")
    
    # 添加条件边 - 直接到结果处理节点
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "process_results",
            "end": "memory_store"
        }
    )
    
    # 添加结果处理到Agent的边
    workflow.add_edge("process_results", "agent")

    # 记忆存储后结束
    workflow.add_edge("memory_store", END)
    
    # 编译图
    return workflow.compile()

# 创建全局Agent图实例
agent_graph = create_agent_graph()
