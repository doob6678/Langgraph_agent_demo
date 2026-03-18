import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# 加载 .env 环境变量
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "doob67nevergiveup")
MYSQL_DB = os.getenv("MYSQL_DB", "agent_memory_db_demo_01")

# 使用 pymysql 作为驱动 (同步驱动，用于当前基础架构，生产中若全异步可换 aiomysql)
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"

try:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_pre_ping=True,      # 每次连接前测试连接是否有效
        pool_recycle=3600,       # 连接重置时间
        pool_size=5,             # 连接池大小
        max_overflow=10          # 超过连接池大小外最多创建的连接
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    logger.error(f"数据库连接初始化失败: {str(e)}")
    raise

def get_db():
    """
    依赖注入：获取数据库会话
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
