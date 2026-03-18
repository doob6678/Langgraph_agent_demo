
import asyncio
import logging
import sys
import os
import uuid

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from langchain_core.messages import HumanMessage
from backend.agent.graph_new_real import agent_graph
from backend.agent.node_ex.memory_node import get_memory_manager
from backend.agent.memory_ex.database import SessionLocal
from backend.agent.memory_ex.models import MemoryContent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_agent_conversation():
    """
    测试 Agent 的记忆能力：
    1. 用户告知信息
    2. Agent 存储记忆
    3. 用户询问相关信息
    4. Agent 能够准确回忆
    5. 测试图片记忆
    """
    
    # 模拟用户
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"=== 开始 Agent 记忆测试 (User: {user_id}) ===")
    
    # 获取 MemoryManager 实例用于验证
    memory_manager = get_memory_manager()

    # 第一轮：告知信息
    logger.info("\n--- 第一轮对话：告知职业 ---")
    inputs_1 = {
        "messages": [HumanMessage(content="你好，我叫李四，我是一名资深的 Java 架构师。请记住我的职业。")],
        "user_id": user_id,
        "user_input": "你好，我叫李四，我是一名资深的 Java 架构师。请记住我的职业。" # AgentState needs user_input
    }
    
    result_1 = await agent_graph.ainvoke(inputs_1)
    response_1 = result_1["answer"]
    logger.info(f"Agent 回复: {response_1}")
    
    # 验证是否存入长期记忆 (因为有“记住”关键词，或者只是存入短期记忆)
    # 我们的策略是：所有交互存入短期记忆，重要信息存入长期记忆。
    # 这里我们检查短期记忆是否持久化到 MySQL
    from backend.agent.memory_ex.models import ShortTermMemoryModel
    db = SessionLocal()
    try:
        # 查短期记忆
        stm_records = db.query(ShortTermMemoryModel).filter(
            ShortTermMemoryModel.user_id == user_id
        ).all()
        logger.info(f"MySQL 中短期记忆记录数: {len(stm_records)}")
        for r in stm_records:
            logger.info(f"  - [{r.role}] {r.content}")
            
        if len(stm_records) >= 2: # User + AI
            logger.info("[验证成功] 短期记忆已持久化。")
        else:
            logger.warning("[验证警告] 短期记忆记录数量不足。")

    except Exception as e:
        logger.error(f"数据库验证失败: {e}")
    finally:
        db.close()

    # 第二轮：询问信息
    logger.info("\n--- 第二轮对话：询问职业 ---")
    # 注意：在真实的 App 中，我们会把之前的 messages 传进去。
    # 但为了测试 MemoryRecallNode 是否从数据库拉取记忆，我们故意不传之前的 messages，
    # 只传当前的 query，看它能否从 MemoryManager 找回上下文。
    inputs_2 = {
        "messages": [HumanMessage(content="我刚才说我是做什么的？")],
        "user_id": user_id,
        "user_input": "我刚才说我是做什么的？"
    }
    
    result_2 = await agent_graph.ainvoke(inputs_2)
    response_2 = result_2["answer"]
    logger.info(f"Agent 回复: {response_2}")
    
    # 验证回复中是否包含职业信息
    if "Java" in response_2 or "架构师" in response_2:
        logger.info("[测试通过] Agent 成功回忆起了用户的职业信息！")
    else:
        logger.error(f"[测试失败] Agent 未能回忆起职业信息。回复内容: {response_2}")
        # 打印一下 context 看看
        logger.info(f"Memory Context: {result_2.get('memory_context', 'N/A')}")

    # 第三轮：图片记忆测试
    logger.info("\n--- 第三轮：图片记忆测试 ---")
    
    # 3.1 模拟存储图片资产
    image_desc = "一张包含蓝色大海和白色沙滩的风景照片"
    image_uri = "oss://images/vacation/sea_001.jpg"
    logger.info(f"模拟上传图片: {image_desc} ({image_uri})")
    
    await memory_manager.store_image_asset(
        user_id=user_id,
        description=image_desc,
        image_uri=image_uri
    )
    
    # 3.2 用户询问图片相关内容
    inputs_3 = {
        "messages": [HumanMessage(content="我之前是不是上传过一张关于大海的照片？")],
        "user_id": user_id,
        "user_input": "我之前是不是上传过一张关于大海的照片？"
    }
    
    result_3 = await agent_graph.ainvoke(inputs_3)
    response_3 = result_3["answer"]
    logger.info(f"Agent 回复: {response_3}")
    
    if "大海" in response_3 or "照片" in response_3 or "上传" in response_3:
         logger.info("[测试通过] Agent 成功回忆起了图片信息！")
    else:
         logger.warning(f"[测试可能失败] Agent 回复可能未包含图片信息: {response_3}")
         logger.info(f"Memory Context: {result_3.get('memory_context', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(test_agent_conversation())
