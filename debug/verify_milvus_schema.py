import os
from dotenv import load_dotenv

try:
    from pymilvus import connections, Collection, utility
except Exception as e:
    raise RuntimeError(f"pymilvus import failed: {e}") from e


def main() -> None:
    load_dotenv()
    alias = "verify_milvus_schema"
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")
    user = os.getenv("MILVUS_USER", "root")
    password = os.getenv("MILVUS_PASSWORD", "")
    db_name = os.getenv("MILVUS_DB", "default")
    image_collection = (os.getenv("IMAGE_MEMORY_COLLECTION") or os.getenv("MILVUS_COLLECTION") or "agent_image_memory").strip()
    long_term_collection = (os.getenv("LONG_TERM_MEMORY_COLLECTION") or "agent_long_term_memory").strip()
    collections = [image_collection, long_term_collection]

    if not connections.has_connection(alias):
        connections.connect(
            alias=alias,
            host=host,
            port=port,
            user=user,
            password=password,
            db_name=db_name,
        )

    for name in collections:
        if not utility.has_collection(name, using=alias):
            print(f"{name}: missing")
            continue
        coll = Collection(name, using=alias)
        fields = [f.name for f in coll.schema.fields]
        has_tenant = "tenant_id" in fields
        print(f"{name}: tenant_id_present={str(has_tenant).lower()} fields={','.join(fields)}")


if __name__ == "__main__":
    main()
