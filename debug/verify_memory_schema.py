import os
from dotenv import load_dotenv
import pymysql


def main() -> None:
    load_dotenv()
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DB", "agent_memory_db_demo_01"),
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = 'memory_contents'
                  AND column_name = 'tenant_id'
                """
            )
            count = int(cur.fetchone()[0])
            print(f"tenant_column_count={count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
