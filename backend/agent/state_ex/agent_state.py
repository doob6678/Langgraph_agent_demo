from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    messages: List[Any]
    user_input: str = ""
    image_data: Optional[bytes] = None
    image_filename: Optional[str] = None
    tool_flags: List[bool] = None
    top_k: int = 5
    answer: str = ""
    images: List[Dict[str, Any]] = None
    search_results: List[Dict[str, Any]] = None
    timing: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    tool_calls: List[Dict[str, Any]] = None
    needs_tool: bool = False
    current_tool: str = ""
    tool_results: Dict[str, Any] = None

    def __post_init__(self):
        if self.tool_flags is None:
            self.tool_flags = [True, True]
        else:
            try:
                flags = list(self.tool_flags) if isinstance(self.tool_flags, list) else []
            except Exception:
                flags = []
            flags = [bool(x) for x in flags]
            if len(flags) < 2:
                flags = (flags + [True, True])[:2]
            self.tool_flags = flags[:2]
        if self.images is None:
            self.images = []
        if self.search_results is None:
            self.search_results = []
        if self.timing is None:
            self.timing = {}
        if self.metadata is None:
            self.metadata = {}
        if self.tool_calls is None:
            self.tool_calls = []
        if self.tool_results is None:
            self.tool_results = {}

    @property
    def use_rag(self) -> bool:
        try:
            return bool((self.tool_flags or [True, True])[0])
        except Exception:
            return True

    @use_rag.setter
    def use_rag(self, v: Any) -> None:
        flags = list(self.tool_flags) if isinstance(self.tool_flags, list) else [True, True]
        while len(flags) < 2:
            flags.append(True)
        flags[0] = bool(v)
        self.tool_flags = flags[:2]

    @property
    def use_search(self) -> bool:
        try:
            return bool((self.tool_flags or [True, True])[1])
        except Exception:
            return True

    @use_search.setter
    def use_search(self, v: Any) -> None:
        flags = list(self.tool_flags) if isinstance(self.tool_flags, list) else [True, True]
        while len(flags) < 2:
            flags.append(True)
        flags[1] = bool(v)
        self.tool_flags = flags[:2]
