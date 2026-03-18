import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent.memory_ex.image_memory import ImageMemory
from backend.agent.memory_ex.long_term_memory import LongTermMemory

try:
    from pymilvus import connections, Collection, utility
except Exception:
    Collection = None
    utility = None
    connections = None


def _detect_image_dim(collection_name: str) -> int:
    if not connections or not utility or not Collection:
        return 0
    alias = "migrate_milvus_schema"
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")
    user = os.getenv("MILVUS_USER", "root")
    password = os.getenv("MILVUS_PASSWORD", "")
    db_name = os.getenv("MILVUS_DB", "default")
    if not connections.has_connection(alias):
        connections.connect(
            alias=alias,
            host=host,
            port=port,
            user=user,
            password=password,
            db_name=db_name,
        )
    if not utility.has_collection(collection_name, using=alias):
        return 0
    coll = Collection(collection_name, using=alias)
    for field in coll.schema.fields:
        if field.name == "embedding":
            dim = getattr(field, "params", {}).get("dim")
            return int(dim or 0)
    return 0


def main() -> None:
    load_dotenv()
    long_term_collection = (os.getenv("LONG_TERM_MEMORY_COLLECTION") or "agent_long_term_memory").strip()
    image_collection = (os.getenv("IMAGE_MEMORY_COLLECTION") or os.getenv("MILVUS_COLLECTION") or "agent_image_memory").strip()
    LongTermMemory()
    image_dim = _detect_image_dim(image_collection)
    image_memory = ImageMemory()
    ok = image_memory.ensure_collection_ready(embedding_dim=image_dim or None)
    print(f"long_term_collection={long_term_collection}")
    print(f"image_collection={image_collection}")
    print(f"image_dim={image_dim}")
    print(f"image_collection_ready={str(ok).lower()}")


if __name__ == "__main__":
    main()
