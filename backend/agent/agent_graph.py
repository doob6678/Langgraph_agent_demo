from typing import TypedDict, List, Annotated
import operator
import logging
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from backend.agent.config_ex.model_config import get_runtime_model_settings
from backend.agent.memory_ex.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

# 定义 Agent 状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: str
    memory_context: str

# 初始化 MemoryManager
# 注意：在实际生产中，MemoryManager 应该是单例或注入的
memory_manager = MemoryManager()

async def memory_recall_node(state: AgentState):
    """
    记忆召回节点：根据用户最新消息检索相关记忆
    """
    logger.info("--- 记忆召回节点 ---")
    messages = state["messages"]
    user_id = state.get("user_id", "default_user")
    
    if not messages:
        return {"memory_context": ""}
        
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return {"memory_context": ""}
        
    query = last_message.content
    logger.info(f"正在检索记忆，Query: {query}")
    
    # 调用 MemoryManager 获取记忆
    # 注意：这里我们使用 recall_context 来获取短期和长期记忆
    memories = await memory_manager.recall_context(user_id=user_id, current_query=query)
    
    short_term = memories.get("short_term", [])
    long_term = memories.get("long_term", [])
    image_memory = memories.get("image_memory", [])
    
    # 格式化记忆上下文
    context_parts = []
    
    if short_term:
        context_parts.append("【短期对话历史】:")
        for m in short_term:
            context_parts.append(f"- {m.get('role')}: {m.get('content')}")
            
    if long_term:
        context_parts.append("\n【长期知识回顾】:")
        for m in long_term:
            content = m.get('content')
            # source = m.get('metadata', {}).get('source', 'milvus')
            # 简化显示
            context_parts.append(f"- [长期记忆] {content}")

    if image_memory:
        context_parts.append("\n【相关图片记忆】:")
        for m in image_memory:
            desc = m.get('content')
            uri = m.get('image_uri', '')
            context_parts.append(f"- [图片] {desc} (URI: {uri})")
            
    memory_context = "\n".join(context_parts)
    logger.info(f"检索到的记忆上下文长度: {len(memory_context)}")
    
    return {"memory_context": memory_context}

async def model_node(state: AgentState):
    """
    模型生成节点：基于记忆上下文和用户输入生成回复
    """
    logger.info("--- 模型生成节点 ---")
    memory_context = state.get("memory_context", "")
    messages = state["messages"]
    last_human_msg = messages[-1].content
    
    # 构建 System Prompt
    system_prompt = f"""你是一个拥有长期记忆的智能助手。
请利用以下记忆上下文来回答用户的问题。如果记忆中有相关信息，请明确引用。

{memory_context}
"""
    
    # 尝试使用真实的 LLM (如果有配置)
    model_settings = get_runtime_model_settings()
    api_key = model_settings.get("api_key", "")
    response_content = ""
    
    if api_key and "your_base_api_key_here" not in api_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_settings.get("base_model") or "doubao-pro-4k",
                api_key=api_key,
                base_url=model_settings.get("base_url") or "https://ark.cn-beijing.volces.com/api/v3"
            )
            response = await llm.ainvoke([SystemMessage(content=system_prompt)] + messages)
            response_content = response.content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            response_content = f"（LLM 调用失败，转为本地逻辑）收到您的消息：{last_human_msg}。根据记忆，我知道：\n{memory_context}"
    else:
        # 如果没有 Key，使用本地逻辑证明记忆存在
        response_content = f"【系统自动回复】\n我已收到您的消息：'{last_human_msg}'。\n\n基于我的【真实数据库记忆】：\n{memory_context}\n\n（注：此回复证明了 Milvus 和 MySQL 的数据已被成功召回并注入到 Context 中。）"

    return {"messages": [AIMessage(content=response_content)]}

async def memory_storage_node(state: AgentState):
    """
    记忆存储节点：保存对话到短期记忆，并判断是否需要存入长期记忆
    """
    logger.info("--- 记忆存储节点 ---")
    messages = state["messages"]
    user_id = state.get("user_id", "default_user")
    
    if len(messages) < 2:
        return {}
        
    last_human = messages[-2]
    last_ai = messages[-1]
    
    if isinstance(last_human, HumanMessage) and isinstance(last_ai, AIMessage):
        await memory_manager.store_interaction(
            user_id=user_id,
            user_query=last_human.content,
            ai_response=last_ai.content,
            is_important=False,
        )
            
    return {}

# 构建 Graph
workflow = StateGraph(AgentState)

workflow.add_node("recall", memory_recall_node)
workflow.add_node("generate", model_node)
workflow.add_node("store", memory_storage_node)

workflow.set_entry_point("recall")
workflow.add_edge("recall", "generate")
workflow.add_edge("generate", "store")
workflow.add_edge("store", END)

agent_app = workflow.compile()
