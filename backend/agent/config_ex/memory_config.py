import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_IMAGE_COLLECTION_NAME = "agent_image_memory"
_CONFIG_FILE = Path(__file__).with_name("memory_config.json")


def _read_config_file() -> Dict[str, Any]:
    if not _CONFIG_FILE.exists():
        return {"image_memory": {"collection_name": DEFAULT_IMAGE_COLLECTION_NAME}}
    try:
        with _CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"image_memory": {"collection_name": DEFAULT_IMAGE_COLLECTION_NAME}}


def get_runtime_memory_settings() -> Dict[str, Any]:
    cfg = _read_config_file()
    image_cfg = cfg.get("image_memory")
    if not isinstance(image_cfg, dict):
        image_cfg = {}
    collection_name = str(image_cfg.get("collection_name") or "").strip() or DEFAULT_IMAGE_COLLECTION_NAME
    return {
        "image_collection_name": collection_name,
        "config_file": str(_CONFIG_FILE),
    }
