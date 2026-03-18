import asyncio
import logging
from backend.agent.memory_ex.long_term_memory import LongTermMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def test_db_operations():
    # 测试长期记忆模块
    ltm = LongTermMemory()
    user_id = "test_user_db_01"

    if not ltm.milvus_collection:
        logging.warning("Milvus 未就绪，跳过长期记忆写入验证。")
        return

    long_content = "A" * 1300
    logging.info(f"准备插入长文本，长度: {len(long_content)}")

    mem_id = await ltm.add_memory(user_id=user_id, content=long_content)
    logging.info(f"插入完成，生成的 memory_id: {mem_id}")

    records = await ltm.list_memories(user_id=user_id, limit=20, visibility="private")
    target = next((item for item in records if item.get("id") == mem_id), None)
    assert target is not None, "Milvus 中未找到新增长期记忆"
    stored_content = str(target.get("content") or "")
    assert len(stored_content) == 1024, f"长期记忆长度未限制为1024，实际: {len(stored_content)}"
    assert stored_content == long_content[:1024], "长期记忆截断内容不符合预期"
    logging.info("验证成功：长期记忆只写入 Milvus，且长度限制为 1024。")

    del_success = await ltm.delete_memory(user_id=user_id, memory_id=mem_id)
    assert del_success, "删除长期记忆失败"
    logging.info("验证成功：长期记忆删除完成。")

if __name__ == "__main__":
    asyncio.run(test_db_operations())
