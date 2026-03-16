from langgraph.graph import StateGraph, END
from backend.agent.state_ex.agent_state import AgentState
from backend.agent.node_ex.agent_node_async import agent_node_async
from backend.agent.tool_ex.state_tool_runner import process_tool_results

def create_async_agent_graph():
    """创建异步Agent图"""
    
    def should_continue(state: AgentState) -> str:
        """决定是否继续执行"""
        if state.needs_tool:
            return "continue"
        if state.answer:
            return "end"
        return "continue"
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("agent", agent_node_async)
    workflow.add_node("process_results", process_tool_results)
    
    # 设置入口点
    workflow.set_entry_point("agent")
    
    # 添加条件边
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

async_agent_graph = create_async_agent_graph()
