import os
from typing import Any, Dict, Optional


def configure_ark(api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    k = (api_key or "").strip() if api_key is not None else (os.getenv("ARK_API_KEY") or "").strip()
    u = (base_url or "").strip() if base_url is not None else (os.getenv("ARK_BASE_URL") or os.getenv("ARK_API_BASE_URL") or "").strip()
    m = (model or "").strip() if model is not None else (os.getenv("ARK_MODEL") or os.getenv("ARK_MODEL_NAME") or "").strip()

    if u:
        os.environ["ARK_BASE_URL"] = u
    if m:
        os.environ["ARK_MODEL"] = m
    if api_key is not None:
        if k:
            os.environ["ARK_API_KEY"] = k
        else:
            try:
                os.environ.pop("ARK_API_KEY", None)
            except Exception:
                pass

    key_set = bool((os.getenv("ARK_API_KEY") or "").strip())
    key_val = (os.getenv("ARK_API_KEY") or "").strip()
    masked = (key_val[:6] + "..." + key_val[-4:]) if key_val and len(key_val) >= 12 else ("set" if key_set else "")
    return {
        "ark_api_key": masked,
        "ark_base_url": (os.getenv("ARK_BASE_URL") or "").strip(),
        "ark_model": (os.getenv("ARK_MODEL") or "").strip(),
        "key_set": key_set,
    }
