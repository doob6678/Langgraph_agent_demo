import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_BASE_MODEL = "doubao-seed-2-0-lite-260215"
_CONFIG_FILE = Path(__file__).resolve().with_name("model_config.json")


def _default_config() -> Dict[str, Any]:
    return {
        "llm": {
            "provider": "openai_compatible",
            "base_url": DEFAULT_BASE_URL,
            "base_model": DEFAULT_BASE_MODEL,
            "api_key_env": "BASE_API_KEY",
        }
    }


def _read_config_file() -> Dict[str, Any]:
    if not _CONFIG_FILE.exists():
        data = _default_config()
        _write_config_file(data)
        return data
    try:
        data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _default_config()
        return data
    except Exception:
        return _default_config()


def _write_config_file(config_data: Dict[str, Any]) -> None:
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(config_data, ensure_ascii=False, indent=2), encoding="utf-8")


def _mask_key(key: str) -> str:
    k = (key or "").strip()
    if not k:
        return ""
    if len(k) >= 12:
        return f"{k[:6]}...{k[-4:]}"
    return "set"


def _get_llm_section(config_data: Dict[str, Any]) -> Dict[str, Any]:
    llm = config_data.get("llm")
    if not isinstance(llm, dict):
        llm = {}
    return llm


def get_runtime_model_settings() -> Dict[str, str]:
    config_data = _read_config_file()
    llm = _get_llm_section(config_data)

    provider = (os.getenv("BASE_PROVIDER") or llm.get("provider") or "openai_compatible").strip()
    base_url = (
        os.getenv("BASE_URL")
        or llm.get("base_url")
        or os.getenv("ARK_BASE_URL")
        or os.getenv("ARK_API_BASE_URL")
        or DEFAULT_BASE_URL
    ).strip()
    base_model = (
        os.getenv("BASE_MODEL")
        or llm.get("base_model")
        or os.getenv("ARK_MODEL")
        or os.getenv("ARK_MODEL_NAME")
        or DEFAULT_BASE_MODEL
    ).strip()

    api_key_env = (llm.get("api_key_env") or "BASE_API_KEY").strip() or "BASE_API_KEY"
    api_key = (
        os.getenv(api_key_env)
        or os.getenv("BASE_API_KEY")
        or os.getenv("ARK_API_KEY")
        or ""
    ).strip()

    return {
        "provider": provider,
        "base_url": base_url or DEFAULT_BASE_URL,
        "base_model": base_model or DEFAULT_BASE_MODEL,
        "api_key": api_key,
        "api_key_env": api_key_env,
    }


def configure_model(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    config_data = _read_config_file()
    llm = _get_llm_section(config_data)

    if provider is not None:
        llm["provider"] = (provider or "").strip() or "openai_compatible"
    if base_url is not None:
        llm["base_url"] = (base_url or "").strip() or DEFAULT_BASE_URL
    if model is not None:
        llm["base_model"] = (model or "").strip() or DEFAULT_BASE_MODEL
    llm["api_key_env"] = (llm.get("api_key_env") or "BASE_API_KEY").strip() or "BASE_API_KEY"
    config_data["llm"] = llm
    _write_config_file(config_data)

    if provider is not None:
        os.environ["BASE_PROVIDER"] = (llm.get("provider") or "openai_compatible").strip() or "openai_compatible"
    if base_url is not None:
        os.environ["BASE_URL"] = (llm.get("base_url") or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    if model is not None:
        os.environ["BASE_MODEL"] = (llm.get("base_model") or DEFAULT_BASE_MODEL).strip() or DEFAULT_BASE_MODEL

    if api_key is not None:
        key_val = (api_key or "").strip()
        if key_val:
            os.environ["BASE_API_KEY"] = key_val
        else:
            try:
                os.environ.pop("BASE_API_KEY", None)
            except Exception:
                pass

    final_cfg = get_runtime_model_settings()
    return {
        "base_api_key": _mask_key(final_cfg["api_key"]),
        "base_url": final_cfg["base_url"],
        "base_model": final_cfg["base_model"],
        "provider": final_cfg["provider"],
        "key_set": bool(final_cfg["api_key"]),
        "config_file": str(_CONFIG_FILE),
    }
