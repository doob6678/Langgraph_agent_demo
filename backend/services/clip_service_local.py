import io
import os
import threading
from dataclasses import dataclass
from typing import List, Sequence, Union, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from PIL import Image
try:
    from modelscope.hub.snapshot_download import snapshot_download
    from modelscope.models.multi_modal.clip.model import CLIPForMultiModalEmbedding
    _HAS_MODELSCOPE = True
except Exception:
    snapshot_download = None
    CLIPForMultiModalEmbedding = None
    _HAS_MODELSCOPE = False


@dataclass(frozen=True)
class _Embedder:
    model: CLIPForMultiModalEmbedding
    image_size: int
    text_max_len: int


class CLIPService:
    """CLIP模型服务（ModelScope CLIPForMultiModalEmbedding）"""

    def __init__(self, model_ref: Optional[str] = None, device: Optional[str] = None):
        self.model_ref = (model_ref or os.getenv("CLIP_MODEL") or "damo/multi-modal_clip-vit-large-patch14_zh").strip()
        self.device = (device or os.getenv("CLIP_DEVICE") or "cpu").strip()
        self._embedder: Optional[_Embedder] = None
        self._lock = threading.RLock()

    def _select_device(self) -> str:
        dev = (self.device or "cpu").strip().lower()
        if dev == "cpu":
            return "cpu"
        if dev.startswith("cuda"):
            if not torch.cuda.is_available():
                return "cpu"
            if ":" in dev:
                try:
                    idx = int(dev.split(":", 1)[1])
                except Exception:
                    idx = 0
            else:
                idx = 0
            return f"cuda:{idx}"
        return "cpu"

    def _text_max_len(self) -> int:
        raw = os.getenv("TEXT_MAX_LEN", "64")
        try:
            n = int((raw or "").strip() or "64")
        except Exception:
            n = 64
        if n < 2:
            n = 2
        if n > 512:
            n = 512
        return n

    def _resolve_model_dir(self) -> str:
        m = (self.model_ref or "").strip()
        if not m:
            raise RuntimeError("empty CLIP_MODEL")
        if os.path.isdir(m):
            return m
        cache_dir = (os.getenv("MODELSCOPE_CACHE_DIR") or "").strip()
        if cache_dir:
            local_candidate = os.path.join(cache_dir, *m.split("/"))
            if os.path.isdir(local_candidate):
                return local_candidate
        return snapshot_download(m)

    def _init_embedder(self) -> _Embedder:
        if self._embedder is not None:
            return self._embedder
        with self._lock:
            if self._embedder is not None:
                return self._embedder

            model_dir = self._resolve_model_dir()
            model = CLIPForMultiModalEmbedding(model_dir=model_dir)
            device = self._select_device()
            if device == "cpu":
                model.device = "cpu"
                model.clip_model.to("cpu")
                model.clip_model.float()
            else:
                model.device = device
                model.clip_model.to(device)

            res = int(getattr(model, "model_info", {}).get("image_resolution", 224) or 224)
            if res <= 0:
                res = 224
            self._embedder = _Embedder(model=model, image_size=res, text_max_len=self._text_max_len())
            return self._embedder

    def _tokenize_texts(self, tokenizer: object, texts: Sequence[str], max_len: int) -> torch.Tensor:
        pad_id = int(tokenizer.vocab["[PAD]"])
        cls_id = int(tokenizer.vocab["[CLS]"])
        sep_id = int(tokenizer.vocab["[SEP]"])

        out: List[List[int]] = []
        for t in texts:
            s = (t or "").strip()
            if not s:
                ids = [cls_id, sep_id]
            else:
                toks = tokenizer.tokenize(s)
                if max_len > 2:
                    toks = toks[: max_len - 2]
                toks = ["[CLS]"] + toks + ["[SEP]"]
                ids = tokenizer.convert_tokens_to_ids(toks)
            if len(ids) < max_len:
                ids = ids + [pad_id] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            out.append(ids)
        return torch.tensor(out, dtype=torch.long)

    def _resize_center_crop(self, img: Image.Image, size: int) -> Image.Image:
        w, h = img.size
        if w <= 0 or h <= 0:
            raise RuntimeError("invalid image size")
        scale = float(size) / float(min(w, h))
        nw = max(size, int(round(w * scale)))
        nh = max(size, int(round(h * scale)))
        img = img.resize((nw, nh), resample=Image.BICUBIC)
        left = int((nw - size) // 2)
        top = int((nh - size) // 2)
        return img.crop((left, top, left + size, top + size))

    def _image_to_tensor(self, img: Image.Image, size: int) -> torch.Tensor:
        img = img.convert("RGB")
        img = self._resize_center_crop(img, size)
        arr = np.asarray(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1)
        t = (t - mean) / std
        return t

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        cleaned = [(t or "").strip() for t in texts]
        if not any(cleaned):
            raise ValueError("empty text")

        emb = self._init_embedder()
        tokens = self._tokenize_texts(emb.model.tokenizer, cleaned, emb.text_max_len)
        dev = torch.device(emb.model.device if getattr(emb.model, "device", None) else "cpu")
        tokens = tokens.to(dev)
        with self._lock, torch.no_grad():
            out = emb.model.forward({"text": tokens})
        vec = out.get("text_embedding") if isinstance(out, dict) else None
        if vec is None and isinstance(out, dict):
            vec = out.get("text_emb")
        if vec is None or not isinstance(vec, torch.Tensor):
            raise RuntimeError("text embedding not found in modelscope output")
        f = vec.detach().to("cpu").to(torch.float32)
        f = torch.nn.functional.normalize(f, dim=-1)
        return f.numpy()

    def encode_image(self, image_data: Union[bytes, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, np.ndarray):
            img = Image.fromarray(image_data)
        else:
            img = image_data

        emb = self._init_embedder()
        t = torch.stack([self._image_to_tensor(img, emb.image_size)], dim=0)
        dev = torch.device(emb.model.device if getattr(emb.model, "device", None) else "cpu")
        t = t.to(dev)
        with self._lock, torch.no_grad():
            out = emb.model.forward({"img": t})
        vec = out.get("img_embedding") if isinstance(out, dict) else None
        if vec is None and isinstance(out, dict):
            vec = out.get("image_embedding")
        if vec is None or not isinstance(vec, torch.Tensor):
            raise RuntimeError("image embedding not found in modelscope output")
        f = vec.detach().to("cpu").to(torch.float32)
        f = torch.nn.functional.normalize(f, dim=-1)
        return f.numpy()

    def compute_similarity(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        return np.dot(text_features, image_features.T)

    def is_loaded(self) -> bool:
        return self._embedder is not None

    def get_model_info(self) -> dict:
        emb = self._init_embedder()
        try:
            dim = int(getattr(emb.model, "model_info", {}).get("embed_dim") or 0)
        except Exception:
            dim = 0
        if dim <= 0:
            try:
                v = self.encode_text("测试")
                dim = int(v.shape[-1])
            except Exception:
                dim = 0
        return {
            "model_ref": self.model_ref,
            "device": self._select_device(),
            "image_size": emb.image_size,
            "text_max_len": emb.text_max_len,
            "embed_dim": dim,
        }


if (os.getenv("CLIP_USE_MOCK") or "").strip().lower() in ("1", "true", "yes", "y"):
    raise RuntimeError("CLIP_USE_MOCK 已被禁用，禁止使用任何 Mock CLIP 实现。")
if not _HAS_MODELSCOPE:
    raise RuntimeError("ModelScope CLIP 依赖不可用，无法初始化真实 CLIP 服务。")

clip_service = CLIPService(
    model_ref=os.getenv("CLIP_MODEL", None),
    device=os.getenv("CLIP_DEVICE", None),
)
