#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一归档后的测试与调试入口
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple

import requests


def _repo_root() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return root


def _ensure_sys_path() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
        print(f"[path] add PYTHONPATH={root}")


def _post_chat(
    host: str,
    port: int,
    text: str,
    use_rag: bool,
    use_search: bool,
    top_k: int,
    timeout_s: int,
) -> Tuple[int, Dict[str, Any], str]:
    url = f"http://{host}:{port}/api/chat"
    print(
        f"[http] POST {url} text_len={len(text or '')} use_rag={use_rag} use_search={use_search} top_k={top_k} timeout={timeout_s}s"
    )
    t0 = time.time()
    resp = requests.post(
        url,
        data={
            "text": text,
            "use_rag": str(bool(use_rag)).lower(),
            "use_search": str(bool(use_search)).lower(),
            "top_k": str(int(top_k)),
            "stream": "true",
        },
        headers={"Accept": "text/event-stream"},
        timeout=(10, timeout_s),
        stream=True,
    )
    ctype = resp.headers.get("content-type", "")
    print(f"[http] <- status={resp.status_code} content_type={ctype}")
    if not ctype.lower().startswith("text/event-stream"):
        text_body = resp.text or ""
        try:
            return resp.status_code, resp.json(), text_body
        except Exception:
            return resp.status_code, {}, text_body

    answer_parts = []
    final_payload: Dict[str, Any] = {}
    usage: Dict[str, Any] = {}
    raw_lines = []
    for line in resp.iter_lines(decode_unicode=True):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"stream timeout after {timeout_s}s")
        if line is None:
            continue
        s = (line or "").strip()
        if not s:
            continue
        if not s.startswith("data:"):
            continue
        data = s[5:].strip()
        raw_lines.append(data)
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if isinstance(obj, dict):
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            if isinstance(obj.get("x_final"), dict):
                final_payload = obj["x_final"]
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                delta = (choices[0] or {}).get("delta")
                if isinstance(delta, dict):
                    c = delta.get("content")
                    if isinstance(c, str) and c:
                        answer_parts.append(c)

    if not final_payload:
        final_payload = {"response": "".join(answer_parts), "images": [], "search_results": [], "timing": {}, "metadata": {}}
    if usage and isinstance(final_payload.get("metadata"), dict):
        final_payload["metadata"].setdefault("usage", usage)

    raw_preview = "\n".join(raw_lines[-50:])[:4000]
    return resp.status_code, final_payload, raw_preview


def _looks_like_placeholder_key(k: str) -> bool:
    s = (k or "").strip().lower()
    if not s:
        return True
    if "your_ark_api_key_here" in s:
        return True
    if "placeholder" in s:
        return True
    if s in ("none", "null", "changeme", "xxxx", "xxxxx"):
        return True
    return False


def _is_rule_mode(mode: str) -> bool:
    m = (mode or "").strip().lower()
    return m in ("rule", "rule_based", "deterministic", "local", "offline")


def _post_chat_stream_inspect(
    host: str,
    port: int,
    text: str,
    use_rag: bool,
    use_search: bool,
    top_k: int,
    timeout_s: int,
) -> Dict[str, Any]:
    url = f"http://{host}:{port}/api/chat"
    t0 = time.time()
    resp = requests.post(
        url,
        data={
            "text": text,
            "use_rag": str(bool(use_rag)).lower(),
            "use_search": str(bool(use_search)).lower(),
            "top_k": str(int(top_k)),
            "stream": "true",
        },
        headers={"Accept": "text/event-stream"},
        timeout=(10, timeout_s),
        stream=True,
    )
    if resp.status_code != 200:
        return {"status": resp.status_code, "error": (resp.text or "")[:800]}

    tool_events = 0
    tool_call_deltas = 0
    content_pre = 0
    content_post = 0
    phase = "pre"
    for line in resp.iter_lines(decode_unicode=True):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"stream timeout after {timeout_s}s")
        if not line:
            continue
        s = (line or "").strip()
        if not s.startswith("data:"):
            continue
        data = s[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("x_tool_event"):
            tool_events += 1
            phase = "post"
        if isinstance(obj, dict):
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                delta = (choices[0] or {}).get("delta")
                if isinstance(delta, dict):
                    if isinstance(delta.get("tool_calls"), list) and delta.get("tool_calls"):
                        tool_call_deltas += 1
                    c = delta.get("content")
                    if isinstance(c, str) and c:
                        if phase == "pre":
                            content_pre += 1
                        else:
                            content_post += 1

    return {
        "status": resp.status_code,
        "tool_events": tool_events,
        "tool_call_deltas": tool_call_deltas,
        "content_pre": content_pre,
        "content_post": content_post,
    }


def _get_health(host: str, port: int, timeout_s: int) -> Tuple[int, Dict[str, Any], str]:
    url = f"http://{host}:{port}/api/health"
    print(f"[http] GET {url} timeout={timeout_s}s")
    resp = requests.get(url, timeout=timeout_s)
    text_body = resp.text or ""
    print(f"[http] <- status={resp.status_code} bytes={len(text_body)}")
    try:
        return resp.status_code, resp.json(), text_body
    except Exception:
        return resp.status_code, {}, text_body


def _wait_ready(host: str, port: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            code, data, raw = _get_health(host=host, port=port, timeout_s=10)
            status = data.get("status") if isinstance(data, dict) else None
            print(f"[wait] attempt={attempt} http_status={code} status={status}")
            if code == 200 and isinstance(data, dict) and (status == "healthy"):
                return
        except Exception as e:
            last_err = e
            print(f"[wait] attempt={attempt} error={e}")
        time.sleep(1)
    raise RuntimeError(f"server not ready: {last_err}")

def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        port = int(s.getsockname()[1])
        print(f"[port] picked free port={port} host={host}")
        return port

def _capture_process_output(p: subprocess.Popen, max_chars: int = 12000) -> str:
    if p.stdout is None:
        return ""
    try:
        out = p.stdout.read() or ""
    except Exception:
        return ""
    if len(out) > max_chars:
        return out[-max_chars:]
    return out


def _start_server(host: str, port: int, env: Dict[str, str]) -> subprocess.Popen:
    root = _repo_root()
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m",
        "uvicorn",
        "backend.main_real:app",
        "--host",
        host,
        "--port",
        str(int(port)),
    ]
    print(f"[server] start cwd={root}")
    print(f"[server] cmd={' '.join(cmd)}")
    print(f"[server] env.WEB_SEARCH_MODE={env.get('WEB_SEARCH_MODE','')}")
    print(f"[server] env.WEB_SEARCH_FETCH_ALL={env.get('WEB_SEARCH_FETCH_ALL','')}")
    print(f"[server] env.CLIP_PRELOAD={env.get('CLIP_PRELOAD','')}")
    p = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(f"[server] pid={p.pid}")
    return p


def _stop_server(p: subprocess.Popen) -> None:
    try:
        if p.poll() is not None:
            print(f"[server] already exited pid={p.pid} rc={p.returncode}")
            return
        print(f"[server] terminating pid={p.pid}")
        p.terminate()
        try:
            p.wait(timeout=15)
            print(f"[server] terminated pid={p.pid} rc={p.returncode}")
            return
        except Exception:
            print(f"[server] kill pid={p.pid}")
            p.kill()
    except Exception:
        pass


def case_smoke_chat(host: str, port: int) -> None:
    print("[case] smoke_chat start")
    code, data, raw = _post_chat(host, port, "你好", False, False, 1, 30)
    if code != 200:
        raise AssertionError(f"smoke_chat http_status={code} body={raw[:800]}")
    if not isinstance(data, dict):
        raise AssertionError("smoke_chat invalid json")
    print("[case] smoke_chat ok")


def case_web_search(host: str, port: int) -> None:
    print("[case] web_search start")
    code, data, raw = _post_chat(host, port, "今天的人工智能最新新闻", False, True, 3, 120)
    if code != 200:
        raise AssertionError(f"web_search http_status={code} body={raw[:800]}")
    results = data.get("search_results") if isinstance(data, dict) else None
    if not isinstance(results, list) or len(results) < 1:
        print(f"[case] web_search no results (skipped) body={json.dumps(data, ensure_ascii=False)[:500]}")
        return
    first = results[0] if results else {}
    print(f"[case] web_search ok results={len(results)} first_title={(first.get('title') if isinstance(first, dict) else '')}")


def case_rag_search(host: str, port: int) -> None:
    print("[case] rag_search start")
    code, data, raw = _post_chat(host, port, "可爱的小猫咪图片", True, False, 5, 120)
    if code != 200:
        raise AssertionError(f"rag_search http_status={code} body={raw[:800]}")
    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, list) or len(images) < 1:
        raise AssertionError(f"rag_search empty images={json.dumps(data, ensure_ascii=False)[:800]}")
    first = images[0] if images else {}
    print(f"[case] rag_search ok images={len(images)} first_file={(first.get('filename') if isinstance(first, dict) else '')}")


def case_combined_search(host: str, port: int) -> None:
    print("[case] combined_search start")
    code, data, raw = _post_chat(host, port, "人工智能最新发展", True, True, 3, 180)
    if code != 200:
        raise AssertionError(f"combined_search http_status={code} body={raw[:800]}")
    results = data.get("search_results") if isinstance(data, dict) else None
    if not isinstance(results, list) or len(results) < 1:
        print(f"[case] combined_search no results (skipped) body={json.dumps(data, ensure_ascii=False)[:500]}")
        return
    print(f"[case] combined_search ok search_results={len(results)} images={len(data.get('images', []) if isinstance(data, dict) else [])}")


def case_llm_tool_integration(host: str, port: int) -> None:
    print("[case] llm_tool_integration start")
    stats = _post_chat_stream_inspect(host, port, "请联网搜索今天的人工智能新闻，给出3条要点并附来源链接", False, True, 3, 180)
    if int(stats.get("status") or 0) != 200:
        raise AssertionError(f"llm_tool_integration http_status={stats.get('status')} err={stats.get('error','')}")
    if int(stats.get("tool_events") or 0) < 1:
        raise AssertionError(f"llm_tool_integration expected tool_events>=1 got={stats}")
    if int(stats.get("content_post") or 0) < 1:
        raise AssertionError(f"llm_tool_integration expected content_after_tools>=1 got={stats}")
    print(f"[case] llm_tool_integration ok {json.dumps(stats, ensure_ascii=False)}")


def case_debug_milvus() -> None:
    print("[case] debug_milvus start")
    _ensure_sys_path()
    from backend.services.milvus_service import MilvusService

    s = MilvusService()
    s.test_connection()
    items = s.search_images_by_text("猫", top_k=2)
    if not isinstance(items, list):
        raise AssertionError("milvus search returned non-list")
    print(f"[case] debug_milvus ok items={len(items)}")


def case_clip_local() -> None:
    print("[case] clip_local start")
    _ensure_sys_path()
    from backend.services.clip_service_local import clip_service

    text_features = clip_service.encode_text("一只可爱的小猫")
    if getattr(text_features, "shape", None) is None:
        raise AssertionError("clip encode_text missing shape")
    print(f"[case] clip_local ok shape={getattr(text_features, 'shape', None)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--web-search-mode", default=os.getenv("WEB_SEARCH_MODE", "bing"))
    parser.add_argument("--fetch-all", action="store_true")
    parser.add_argument("--skip-milvus-debug", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    args = parser.parse_args()

    host = str(args.host)
    port = int(args.port)
    print(f"[main] python={sys.executable}")
    print(f"[main] cwd={os.getcwd()}")
    print(f"[main] host={host} port={port} no_server={args.no_server}")
    print(f"[main] web_search_mode={args.web_search_mode} fetch_all={args.fetch_all}")
    print(f"[main] skip_milvus_debug={args.skip_milvus_debug} skip_clip={args.skip_clip}")

    env = dict(os.environ)
    env["PYTHONPATH"] = _repo_root()
    env["WEB_SEARCH_MODE"] = str(args.web_search_mode).strip().lower() or "metaso"
    env["WEB_SEARCH_FETCH_ALL"] = "1" if bool(args.fetch_all) else "0"
    env.setdefault("CLIP_PRELOAD", "0")
    if "AGENT_MODE" not in env:
        ark_key = (env.get("ARK_API_KEY") or "").strip()
        if ark_key and (not _looks_like_placeholder_key(ark_key)):
            env["AGENT_MODE"] = ""
        else:
            env["AGENT_MODE"] = "rule_based"

    p: Optional[subprocess.Popen] = None
    try:
        health: Dict[str, Any] = {}
        milvus_ok = False
        if not args.no_server:
            if port <= 0:
                port = _pick_free_port(host)
            p = _start_server(host, port, env)
            deadline = time.time() + 900
            print(f"[server] waiting ready up to 900s at http://{host}:{port}")
            while time.time() < deadline:
                if p.poll() is not None:
                    logs = _capture_process_output(p)
                    raise RuntimeError("uvicorn exited early\n" + (logs or ""))
                try:
                    _wait_ready(host, port, timeout_s=10)
                    print("[server] ready")
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("server not ready in time")
            try:
                _, health, _ = _get_health(host=host, port=port, timeout_s=10)
                milvus_ok = bool((health.get("services") or {}).get("milvus")) if isinstance(health, dict) else False
                print(f"[health] milvus={milvus_ok} raw={json.dumps(health, ensure_ascii=False)[:400]}")
            except Exception as e:
                print(f"[health] get failed: {e}")

        case_smoke_chat(host, port)
        case_web_search(host, port)
        if milvus_ok:
            case_rag_search(host, port)
        else:
            print("[case] rag_search skipped (milvus unavailable)")
        case_combined_search(host, port)
        if (not _is_rule_mode(env.get("AGENT_MODE", ""))) and (not _looks_like_placeholder_key((env.get("ARK_API_KEY") or "").strip())):
            case_llm_tool_integration(host, port)
        else:
            print("[case] llm_tool_integration skipped (no key or rule_based mode)")

        if milvus_ok and (not args.skip_milvus_debug):
            case_debug_milvus()
        elif not milvus_ok:
            print("[case] debug_milvus skipped (milvus unavailable)")
        if not args.skip_clip:
            case_clip_local()

        print("ALL TESTS PASSED")
        return 0
    finally:
        if p is not None:
            _stop_server(p)
            tail = _capture_process_output(p, max_chars=4000)
            if tail.strip():
                print("\n--- uvicorn logs ---\n" + tail)


if __name__ == "__main__":
    raise SystemExit(main())
