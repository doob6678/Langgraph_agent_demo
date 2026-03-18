import asyncio
import logging
from backend.agent.memory_ex.database import engine, SessionLocal
from backend.agent.memory_ex.models import MemoryContent, Base
from backend.agent.memory_ex.long_term_memory import LongTermMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def test_db_operations():
    # 确保表结构创建
    Base.metadata.create_all(bind=engine)
    logging.info("MySQL 表结构初始化完成")

    # 测试长期记忆模块
    ltm = LongTermMemory()
    user_id = "test_user_db_01"
    
    # 插入超长文本 (触发 MySQL 存储，因为设置了 len > 50)
    long_content = "这是一段非常非常长的文本，用来测试超过长度限制后是否会被正确存储到 MySQL 的 memory_contents 扩展表中。" * 5
    logging.info(f"准备插入长文本，长度: {len(long_content)}")
    
    mem_id = await ltm.add_memory(user_id=user_id, content=long_content)
    logging.info(f"插入完成，生成的 memory_id: {mem_id}")
    
    # 验证数据库中是否真实存在
    db = SessionLocal()
    try:
        record = db.query(MemoryContent).filter(MemoryContent.id == mem_id).first()
        if record:
            logging.info(f"成功从 MySQL 中读取到记录: id={record.id}")
            assert record.content == long_content, "读取的内容与插入的内容不一致！"
        else:
            logging.error("未能在 MySQL 中找到插入的记录！")
            assert False, "MySQL 插入失败"
    finally:
        db.close()
        
    # 删除测试记录
    del_success = await ltm.delete_memory(user_id=user_id, memory_id=mem_id)
    logging.info(f"删除记录状态: {del_success}")
    
    # 再次验证是否删除
    db = SessionLocal()
    try:
        record_after_del = db.query(MemoryContent).filter(MemoryContent.id == mem_id).first()
        assert record_after_del is None, "记录未能从 MySQL 中成功删除！"
        logging.info("验证成功：记录已从 MySQL 中删除。")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_db_operations())
