from typing import Any, Dict, Optional

from backend.agent.config_ex.model_config import configure_model


def configure_ark(api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    return configure_model(api_key=api_key, base_url=base_url, model=model)
