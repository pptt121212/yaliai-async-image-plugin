# -*- coding: utf-8 -*-
"""
鸭梨 AI 图像生成插件

参数同步：
- main.py 同级 config.json
- 前端 sendAction('save_param') 写入
- 前端 sendAction('load_params') 读取
- generate() 时以 config.json 为最高优先级

支持：
- Yali AI Gateway native Gemini and OpenAI Images endpoints
- Durable async submission, task polling, and local image delivery
- OpenAI Images reference files use multipart uploads; Gemini uses inlineData

"""

import os
import re
import base64
import time
import json
import threading
import tempfile
import uuid
import requests

from requests.adapters import HTTPAdapter

try:
    from urllib3.exceptions import InsecureRequestWarning

    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
except Exception:
    pass

from pathlib import Path
from io import BytesIO
from PIL import Image
from urllib.parse import urljoin, urlparse


_PLUGIN_FILE = __file__
plugin_dir = Path(__file__).parent


# ===================== 自管理配置文件 =====================

_CONFIG_PATH = plugin_dir / "config.json"
_config_lock = threading.Lock()
_ASYNC_TASK_LOG_PATH = plugin_dir / "async_tasks.jsonl"
_async_task_log_lock = threading.Lock()


def _load_config():
    with _config_lock:
        if not _CONFIG_PATH.exists():
            return {}

        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data if isinstance(data, dict) else {}

        except Exception as e:
            print(f"[Yali AI Image] 读取配置失败: {e}")
            return {}


def _save_config(params):
    if not isinstance(params, dict):
        return False

    params = dict(params)
    if "model" in params:
        params["model"] = _normalize_model(params.get("model"))

    with _config_lock:
        try:
            existing = {}

            if _CONFIG_PATH.exists():
                try:
                    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                        existing = json.load(f)

                    if not isinstance(existing, dict):
                        existing = {}
                except Exception:
                    existing = {}

            existing.update(params)
            for retired_key in _RETIRED_PARAM_KEYS:
                existing.pop(retired_key, None)
            _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"[Yali AI Image] 保存配置失败: {e}")
            return False


def _save_single_param(key, value):
    if key is None:
        return False

    return _save_config({key: value})


# ===================== 默认参数 =====================

_default_params = {
    "gpt_api_key": "",
    "gemini_api_key": "",
    "endpoint": "https://api.yaliai.com",
    "model": "gemini-3.1-flash-image-preview",
    "aspect_ratio": "16:9",
    "image_size": "4K",
    "quality": "medium",
}

# These are execution guarantees, not end-user tuning knobs. A gateway image
# request can legitimately take minutes, while polling is cheap and cancellable.
_GATEWAY_HTTP_TIMEOUT_SECONDS = 300
_IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 300
_ASYNC_INITIAL_DELAY_SECONDS = 30
_ASYNC_POLL_INTERVAL_SECONDS = 5
_ASYNC_MAX_WAIT_SECONDS = 1800
_RETIRED_PARAM_KEYS = {
    "api_key",
    "request_timeout",
    "download_timeout",
    "async_initial_delay",
    "async_poll_interval",
    "async_max_wait",
    "retry_count",
}


def _ensure_config_exists():
    if not _CONFIG_PATH.exists():
        ok = _save_config(_default_params)
        print(f"[Yali AI Image] {'已创建默认配置文件' if ok else '创建默认配置文件失败'}: {_CONFIG_PATH}")
        return

    # Remove retired UI controls once, while preserving explicit user choices.
    current = _load_config()
    migration = {}
    legacy_key = str(current.get("api_key", "") or "").strip()
    if legacy_key:
        if not str(current.get("gpt_api_key", "") or "").strip():
            migration["gpt_api_key"] = legacy_key
        if not str(current.get("gemini_api_key", "") or "").strip():
            migration["gemini_api_key"] = legacy_key
        migration["api_key"] = ""
    for key, value in _default_params.items():
        if key not in current:
            migration[key] = value
    normalized_model = _normalize_model(current.get("model"))
    if current.get("model") != normalized_model:
        migration["model"] = normalized_model
    if migration or any(key in current for key in _RETIRED_PARAM_KEYS):
        _save_config(migration)


# ===================== 模型配置 =====================

AVAILABLE_MODELS = [
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gpt-image-2",
]

GEMINI_MODELS = {
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
}

GPT_IMAGE_MODELS = {
    "gpt-image-2",
}

ASPECT_RATIOS = [
    "auto",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "5:4",
    "4:5",
    "21:9",
]

IMAGE_SIZES = [
    "1K",
    "2K",
    "4K",
]


# ===================== OpenAI Images 尺寸映射 =====================
"""
只影响 OpenAI Images。

Gemini / 香蕉模型不使用这里的 GPT_IMAGE_SIZE_MAP。
Gemini 仍然按原逻辑把 imageSize 原样传给 generationConfig.imageConfig。

完整映射表来自用户提供截图：

比例      1K 标准       2K 高清       4K 超清
1:1      1024x1024    2048x2048    2880x2880
2:3      688x1024     1360x2048    2336x3520
3:2      1024x688     2048x1360    3520x2336
3:4      768x1024     1536x2048    2480x3312
4:3      1024x768     2048x1536    3312x2480
9:16     608x1088     1152x2048    2160x3840
16:9     1088x608     2048x1152    3840x2160
21:9     1248x528     2048x880     3840x1648

注意：
- 表格没有 5:4 / 4:5。
- 如果 OpenAI Images 选择 5:4 / 4:5，会提示不支持。
"""

GPT_IMAGE_SIZE_MAP = {
    "1K": {
        "1:1": "1024x1024",
        "2:3": "688x1024",
        "3:2": "1024x688",
        "3:4": "768x1024",
        "4:3": "1024x768",
        "9:16": "608x1088",
        "16:9": "1088x608",
        "21:9": "1248x528",
    },
    "2K": {
        "1:1": "2048x2048",
        "2:3": "1360x2048",
        "3:2": "2048x1360",
        "3:4": "1536x2048",
        "4:3": "2048x1536",
        "9:16": "1152x2048",
        "16:9": "2048x1152",
        "21:9": "2048x880",
    },
    "4K": {
        "1:1": "2880x2880",
        "2:3": "2336x3520",
        "3:2": "3520x2336",
        "3:4": "2480x3312",
        "4:3": "3312x2480",
        "9:16": "2160x3840",
        "16:9": "3840x2160",
        "21:9": "3840x1648",
    },
}


# ===================== 全局参数 =====================

_global_params = _default_params.copy()
_global_params.update(_load_config())


# ===================== 参数工具 =====================

def _clean_empty_credentials(src):
    if not isinstance(src, dict):
        return {}

    data = dict(src)

    for key in ("api_key", "gpt_api_key", "gemini_api_key"):
        if not str(data.get(key, "") or "").strip():
            data.pop(key, None)

    if not str(data.get("endpoint", "") or "").strip():
        data.pop("endpoint", None)

    return data


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _prompt_preview(value, limit=120):
    """Keep only a bounded, Unicode-safe prompt hint for local task logs."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_endpoint(endpoint):
    endpoint = str(endpoint or "").strip()

    if not endpoint:
        endpoint = _default_params["endpoint"]

    return endpoint.rstrip("/")


def _normalize_model(model):
    model = str(model or "").strip()

    if not model:
        return _default_params["model"]

    compact = re.sub(r"[^a-z0-9]", "", model.lower())
    # Older tool settings used names such as gpt-image2-Pro. They are still
    # OpenAI Images selections and must never enter the Gemini endpoint branch.
    if compact.startswith("gptimage2") or compact.startswith("openaiimage"):
        return "gpt-image-2"

    return model


def _api_key_for_model(params, model):
    """Select the model-family credential while accepting a pre-v3 config."""
    key_name = "gpt_api_key" if model in GPT_IMAGE_MODELS else "gemini_api_key"
    api_key = str(params.get(key_name, "") or "").strip()
    if api_key:
        return api_key
    return str(params.get("api_key", "") or "").strip()


def _normalize_reference_images(reference_images):
    if not reference_images:
        return {}

    if isinstance(reference_images, dict):
        return reference_images

    if isinstance(reference_images, list):
        return {idx: value for idx, value in enumerate(reference_images)}

    return {}


def _collect_valid_reference_images(reference_images, max_images=8):
    reference_images = _normalize_reference_images(reference_images)

    local_paths = []
    urls = []

    for _, value in reference_images.items():
        if not value:
            continue

        text = str(value).strip()

        if not text:
            continue

        lower = text.lower()

        if lower.startswith("http://") or lower.startswith("https://"):
            urls.append(text)
        elif os.path.exists(text) and os.path.getsize(text) > 0:
            local_paths.append(os.path.abspath(text))

        if len(local_paths) + len(urls) >= max_images:
            break

    return {
        "local": local_paths[:max_images],
        "urls": urls[:max_images],
    }


def _merge_runtime_params(context):
    """
    优先级：
    默认参数 < _global_params < config.json < 当前调用的 context.plugin_params
    """
    merged = _default_params.copy()

    merged.update(_clean_empty_credentials(_global_params))

    file_params = _clean_empty_credentials(_load_config())
    merged.update(file_params)

    ctx_params = context.get("plugin_params")

    # The host's current invocation is authoritative. This allows one plugin
    # instance to switch between the two gateway interface types without a
    # stale persisted model overriding the request.
    if isinstance(ctx_params, dict):
        merged.update(_clean_empty_credentials(ctx_params))

    merged["model"] = _normalize_model(merged.get("model"))

    try:
        print("[Yali AI Image][PARAM_SRC]")
        print(f"  _global_params.model = {_global_params.get('model')}")
        print(f"  config.json.model = {file_params.get('model')}")
        print(f"  context.plugin_params.model = {ctx_params.get('model') if isinstance(ctx_params, dict) else None}")
        print(f"  merged.model = {merged.get('model')}")
    except Exception:
        pass

    return merged


def get_params():
    params = _default_params.copy()
    params.update(_clean_empty_credentials(_load_config()))
    params["model"] = _normalize_model(params.get("model"))
    return params


# ===================== 插件基础接口 =====================

def get_info():
    return {
        "name": "鸭梨 AI 异步图像生成插件",
        "description": "通过鸭梨 AI 网关异步生成图片，支持 Gemini 原生接口、OpenAI Images 文件上传和任务日志。",
        "version": "3.1.0",
        "author": "Yali AI",
    }


def create_ui(parent_widget=None):
    return str(plugin_dir / "ui" / "index.html")


def load_params(params):
    global _global_params

    if isinstance(params, dict):
        params = _clean_empty_credentials(params)
        if "model" in params:
            params["model"] = _normalize_model(params.get("model"))
        _global_params.update(params)

    print("[Yali AI Image] load_params 已更新内存参数")


def handle_action(action, data=None):
    global _global_params

    if data is None:
        data = {}

    if action == "open_task_logs":
        return {"ok": True, "open_page": "task_log.html"}

    if action == "save_param":
        key = data.get("key")
        value = data.get("value")

        if key is None:
            return {"ok": False, "error": "缺少 key"}

        ok = _save_single_param(key, value)

        if ok:
            if key == "model":
                value = _normalize_model(value)
            _global_params[key] = value

        return {"ok": ok}

    elif action == "save_all_params":
        params = data.get("params", {})

        if not isinstance(params, dict):
            return {"ok": False, "error": "params 必须是 dict"}

        ok = _save_config(params)

        if ok:
            if "model" in params:
                params = dict(params)
                params["model"] = _normalize_model(params.get("model"))
            _global_params.update(params)

        return {"ok": ok}

    elif action == "load_params":
        return {"ok": True, "params": _load_config()}

    elif action == "get_params":
        return {"ok": True, "params": get_params()}

    elif action == "get_config_path":
        return {
            "ok": True,
            "path": str(_CONFIG_PATH),
            "exists": _CONFIG_PATH.exists(),
        }

    elif action == "get_task_logs":
        page = max(1, _safe_int(data.get("page", 1), 1))
        page_size = min(50, max(10, _safe_int(data.get("page_size", 20), 20)))
        status = str(data.get("status", "") or "").strip().lower()
        task_id = str(data.get("task_id", "") or "").strip()
        return {"ok": True, **_query_async_task_logs(page, page_size, status, task_id)}

    elif action == "clear_task_logs":
        mode = str(data.get("mode", "before") or "before").strip().lower()
        if mode == "all":
            removed = _clear_async_task_logs()
        else:
            days = min(3650, max(1, _safe_int(data.get("days", 30), 30)))
            removed = _clear_async_task_logs(before_timestamp=time.time() - days * 86400)
        return {"ok": True, "removed": removed}

    return {"ok": False, "error": f"未知动作: {action}"}


def _record_async_task(event, **fields):
    """Persist a small task receipt without storing prompts or image data."""
    entry = {
        "timestamp": int(time.time()),
        "event": str(event),
        **{str(key): value for key, value in fields.items()},
    }
    with _async_task_log_lock:
        _ASYNC_TASK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_ASYNC_TASK_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_async_task_events():
    if not _ASYNC_TASK_LOG_PATH.exists():
        return []
    events = []
    try:
        with _async_task_log_lock:
            with open(_ASYNC_TASK_LOG_PATH, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(item, dict):
                        events.append(item)
    except OSError:
        return []
    return events


def _summarize_async_task_events(events):
    grouped = {}
    for event in events:
        task_id = str(event.get("task_id", "") or "").strip()
        if not task_id:
            continue
        timestamp = _safe_int(event.get("timestamp", 0), 0)
        summary = grouped.setdefault(task_id, {
            "task_id": task_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "event_count": 0,
            "event": "",
            "status": "",
            "error": "",
            "trace_id": "",
            "request_id": "",
            "query_path": "",
            "output_count": 0,
            "output_urls": [],
            "output_path": "",
            "output_type": "",
            "model": "",
            "quality": "",
            "image_size": "",
            "aspect_ratio": "",
            "request_size": "",
            "protocol": "",
            "viewer_index": 0,
            "unique_name": "",
            "generation_round": 0,
            "output_position": 0,
            "batch_index": 0,
            "batch_num": 1,
            "reference_image_count": 0,
            "prompt_preview": "",
        })
        summary["created_at"] = min(summary["created_at"], timestamp) if timestamp else summary["created_at"]
        summary["updated_at"] = max(summary["updated_at"], timestamp)
        summary["event_count"] += 1
        summary["event"] = str(event.get("event", "") or summary["event"])
        for key in (
            "status", "error", "trace_id", "request_id", "query_path", "output_path", "output_type",
            "model", "quality", "image_size", "aspect_ratio", "request_size", "protocol", "viewer_index", "unique_name",
            "generation_round", "output_position", "batch_index", "batch_num", "reference_image_count", "prompt_preview",
        ):
            if event.get(key) not in (None, ""):
                summary[key] = event[key]
        if event.get("output_count") is not None:
            summary["output_count"] = event.get("output_count")
        for url in event.get("output_urls", []) if isinstance(event.get("output_urls"), list) else []:
            if url and url not in summary["output_urls"]:
                summary["output_urls"].append(url)
        if event.get("output_url") and event["output_url"] not in summary["output_urls"]:
            summary["output_urls"].append(event["output_url"])
    return sorted(grouped.values(), key=lambda item: item["updated_at"], reverse=True)


def _query_async_task_logs(page, page_size, status="", task_id=""):
    tasks = _summarize_async_task_events(_read_async_task_events())
    if task_id:
        tasks = [item for item in tasks if task_id.lower() in item["task_id"].lower()]
    if status:
        tasks = [item for item in tasks if str(item.get("status", "")).lower() == status]
    total = len(tasks)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)
    start = (page - 1) * page_size
    return {
        "tasks": tasks[start:start + page_size],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
    }


def _clear_async_task_logs(before_timestamp=None):
    if not _ASYNC_TASK_LOG_PATH.exists():
        return 0
    with _async_task_log_lock:
        try:
            with open(_ASYNC_TASK_LOG_PATH, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            if before_timestamp is None:
                kept = []
            else:
                kept = []
                for line in lines:
                    try:
                        item = json.loads(line)
                    except (TypeError, ValueError):
                        kept.append(line)
                        continue
                    if _safe_int(item.get("timestamp", 0), 0) >= before_timestamp:
                        kept.append(line)
            removed = len(lines) - len(kept)
            temp_path = _ASYNC_TASK_LOG_PATH.with_suffix(".jsonl.tmp")
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.writelines(kept)
            os.replace(temp_path, _ASYNC_TASK_LOG_PATH)
            return removed
        except OSError:
            return 0


def _new_http_session():
    """Use one bounded connection pool for one host generation invocation."""
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _is_cancelled(context):
    """Accept the host's cancellation primitives without assuming one SDK version."""
    for key in ("cancel_event", "cancellation_event", "stop_event"):
        event = context.get(key)
        if event is not None and callable(getattr(event, "is_set", None)) and event.is_set():
            return True
    for key in ("is_cancelled", "should_cancel"):
        callback = context.get(key)
        if callable(callback):
            try:
                if callback():
                    return True
            except Exception:
                pass
    return bool(context.get("cancelled", False))


def _sleep_with_cancel(seconds, is_cancelled):
    deadline = time.monotonic() + max(0, float(seconds))
    while True:
        if is_cancelled():
            raise Exception("任务已被宿主取消；已提交的鸭梨 AI 异步任务会由网关继续完成")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def _new_request_id(context, output_position):
    unique_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(context.get("unique_name", "image")))[:24]
    generation_round = _safe_int(context.get("generation_round", 0), 0)
    return f"yaliai_plugin_{unique_name}_{generation_round}_{output_position}_{uuid.uuid4().hex}"


def _gateway_origin(endpoint):
    parsed = urlparse(_normalize_endpoint(endpoint))
    if not parsed.scheme or not parsed.netloc:
        raise Exception("PLUGIN_ERROR:::鸭梨 AI Gateway 地址无效")
    return f"{parsed.scheme}://{parsed.netloc}"


def _absolute_gateway_url(endpoint, value):
    value = str(value or "").strip()
    if not value:
        return ""
    return value if value.startswith(("http://", "https://")) else urljoin(_gateway_origin(endpoint) + "/", value.lstrip("/"))


def _gateway_headers(api_key, request_id):
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-Request-ID": request_id,
        "Idempotency-Key": request_id,
    }


def _decode_json_response(response, label):
    try:
        return response.json()
    except ValueError as exc:
        raise Exception(f"{label} 返回非 JSON（HTTP {response.status_code}）: {response.text[:1000]}") from exc


def _submit_async_request(session, url, api_key, request_id, request_timeout, *, json_payload=None, form_data=None, files=None, task_metadata=None):
    headers = _gateway_headers(api_key, request_id)
    if json_payload is not None:
        headers["Content-Type"] = "application/json"
        response = session.post(url, headers=headers, json=json_payload, timeout=request_timeout)
    else:
        response = session.post(url, headers=headers, data=form_data, files=files, timeout=request_timeout)

    payload = _decode_json_response(response, "鸭梨 AI 异步提交")
    if response.status_code != 202:
        message = payload.get("message") if isinstance(payload, dict) else ""
        raise Exception(f"鸭梨 AI 异步提交失败 HTTP {response.status_code}: {message or response.text[:1000]}")

    task_id = str(payload.get("task_id") or "").strip()
    query_path = str(payload.get("query_path") or "").strip()
    if not task_id:
        raise Exception("鸭梨 AI 异步提交响应缺少 task_id")
    if not query_path:
        query_path = f"/v1/image/tasks/{task_id}"

    _record_async_task(
        "accepted",
        task_id=task_id,
        query_path=query_path,
        trace_id=payload.get("trace_id", ""),
        request_id=request_id,
        status=payload.get("status", "queued"),
        provider="yaliai_gateway",
        **(task_metadata or {}),
    )
    return payload


def _extract_async_outputs(payload):
    """Return ordered {type, value} items from public async results."""
    if not isinstance(payload, dict):
        return []

    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    outputs = []
    data = result.get("data") if isinstance(result, dict) else None
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("url"):
                outputs.append({"type": "url", "value": str(item["url"])})
            elif item.get("b64_json"):
                outputs.append({"type": "b64", "value": str(item["b64_json"])})

    if outputs:
        return outputs

    # Native Gemini is normally normalized by the Gateway to data[].url.
    # Keep this fallback for a direct/native response during local testing.
    for candidate in payload.get("candidates", []) if isinstance(payload.get("candidates"), list) else []:
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        for part in content.get("parts", []) if isinstance(content, dict) else []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                outputs.append({"type": "b64", "value": str(inline["data"])})
            file_data = part.get("fileData") or part.get("file_data")
            if isinstance(file_data, dict) and file_data.get("fileUri"):
                outputs.append({"type": "url", "value": str(file_data["fileUri"])})
    return outputs


def _poll_async_task(session, endpoint, api_key, accepted, request_timeout, initial_delay, poll_interval, max_wait, progress, is_cancelled=lambda: False):
    task_id = str(accepted["task_id"])
    query_path = str(accepted.get("query_path") or f"/v1/image/tasks/{task_id}")
    query_url = _absolute_gateway_url(endpoint, query_path)
    _record_async_task("polling", task_id=task_id, query_path=query_path)

    initial_delay = max(30, int(initial_delay))
    poll_interval = max(1, int(poll_interval))
    max_wait = max(60, int(max_wait))
    progress(f"已提交任务，等待 {initial_delay} 秒后开始查询", 15)
    _sleep_with_cancel(initial_delay, is_cancelled)
    deadline = time.monotonic() + max_wait
    poll_count = 0
    last_status = ""

    while time.monotonic() < deadline:
        if is_cancelled():
            _record_async_task("cancelled", task_id=task_id, status="client_cancelled", poll_count=poll_count)
            raise Exception("任务已被宿主取消；已提交的鸭梨 AI 异步任务会由网关继续完成")
        poll_count += 1
        response = session.get(
            query_url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=request_timeout,
        )
        payload = _decode_json_response(response, "鸭梨 AI 任务查询")
        if response.status_code != 200:
            message = payload.get("message") if isinstance(payload, dict) else ""
            raise Exception(f"鸭梨 AI 任务查询失败 HTTP {response.status_code}: {message or response.text[:1000]}")

        status = str(payload.get("status") or "").strip().lower()
        if status != last_status or poll_count == 1:
            _record_async_task("status", task_id=task_id, status=status, poll_count=poll_count)
            last_status = status
        if status == "completed":
            outputs = _extract_async_outputs(payload)
            if not outputs:
                raise Exception("鸭梨 AI 任务已完成，但响应中没有图片 URL 或 Base64 数据")
            _record_async_task(
                "completed",
                task_id=task_id,
                status=status,
                output_count=len(outputs),
                output_urls=[item["value"] for item in outputs if item.get("type") == "url"],
                output_kinds=[item.get("type", "") for item in outputs],
            )
            for item in outputs:
                item["task_id"] = task_id
            progress("任务完成，开始按顺序下载图片", 85)
            return outputs
        if status in {"failed", "cancelled", "canceled", "expired"}:
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            message = error.get("message") or error.get("code") or payload.get("message") or status
            _record_async_task("failed", task_id=task_id, status=status, error=str(message))
            raise Exception(f"鸭梨 AI 任务{status}: {message}")

        elapsed = max_wait - max(0, int(deadline - time.monotonic()))
        progress(f"任务{status or '处理中'}，已等待 {elapsed} 秒", min(84, 20 + int(elapsed * 60 / max(1, max_wait))))
        _sleep_with_cancel(poll_interval, is_cancelled)

    _record_async_task("timeout", task_id=task_id, status="poll_timeout", poll_count=poll_count)
    raise Exception(f"鸭梨 AI 任务轮询超时（task_id={task_id}）")


# ===================== 通用图片工具 =====================

def guess_mime_type(file_path):
    ext = os.path.splitext(str(file_path))[1].lower()

    if ext == ".png":
        return "image/png"

    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"

    if ext == ".webp":
        return "image/webp"

    if ext == ".gif":
        return "image/gif"

    return "image/png"


def _get_output_dir(context):
    output_dir = context.get("output_dir")

    if output_dir:
        return os.path.abspath(output_dir)

    project_path = context.get("project_path")

    if project_path:
        return os.path.abspath(project_path)

    return ""


def _build_output_path(context, output_dir, ext="png", position_override=None):
    viewer_index = int(context.get("viewer_index", 0))
    unique_name = str(context.get("unique_name", "output"))
    generation_round = int(context.get("generation_round", 0))

    output_positions = context.get("output_position", [0])

    if not output_positions:
        output_positions = [0]

    position = output_positions[0] if position_override is None else position_override

    filename = f"{viewer_index:04d}_{unique_name}_{generation_round}_{position}.{ext}"

    return os.path.abspath(os.path.join(output_dir, filename))


def save_image_base64_to_output(image_base64, context, output_dir, position_override=None):
    raw_value = str(image_base64 or "")
    if raw_value.startswith("data:") and "," in raw_value:
        raw_value = raw_value.split(",", 1)[1]
    image_data = base64.b64decode(raw_value)
    output_path = _build_output_path(context, output_dir, "png", position_override)
    with Image.open(BytesIO(image_data)) as image:
        image.load()
        image.save(output_path, "PNG")

    return output_path


def save_image_bytes_to_output(image_bytes, context, output_dir, position_override=None):
    output_path = _build_output_path(context, output_dir, "png", position_override)
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        image.save(output_path, "PNG")

    return output_path


def download_url_to_output(image_url, context, output_dir, download_timeout=300, position_override=None, session=None, is_cancelled=lambda: False):
    temp_path = ""
    owns_session = session is None
    session = session or _new_http_session()
    try:
        # Upstream image hosts may use expired, self-signed, or incomplete
        # certificate chains. The gateway has already selected this URL; image
        # delivery must remain compatible with those upstreams.
        with session.get(image_url, timeout=download_timeout, stream=True, verify=False) as response:
            if response.status_code != 200:
                raise Exception(f"下载图片失败，HTTP {response.status_code}: {response.text[:300]}")
            with tempfile.NamedTemporaryFile(prefix="yaliai_image_", suffix=".download", dir=output_dir, delete=False) as temp:
                temp_path = temp.name
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if is_cancelled():
                        raise Exception("任务已被宿主取消；图片下载已停止")
                    if chunk:
                        temp.write(chunk)
        with Image.open(temp_path) as image:
            image.load()
            output_path = _build_output_path(context, output_dir, "png", position_override)
            image.save(output_path, "PNG")
            return output_path
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        if owns_session:
            session.close()


# ===================== Gemini 原生接口 =====================

def build_gemini_parts(prompt, reference_images):
    parts = []

    if prompt and str(prompt).strip():
        parts.append({"text": str(prompt).strip()})

    reference_images = _normalize_reference_images(reference_images)

    for position, img_path in reference_images.items():
        if not img_path:
            continue

        img_path = str(img_path).strip()

        if not os.path.exists(img_path):
            print(f"⚠️ 参考图不存在，跳过: {img_path}")
            continue

        if os.path.getsize(img_path) <= 0:
            print(f"⚠️ 参考图为空文件，跳过: {img_path}")
            continue

        try:
            with open(img_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            mime_type = guess_mime_type(img_path)

            parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": image_data,
                }
            })

            print(f"添加 Gemini 参考图: {position} -> {img_path} ({mime_type})")

        except Exception as e:
            print(f"加载参考图失败 {img_path}: {e}")
            raise

    return parts


def extract_image_base64_from_gemini_response(response_json, download_timeout=300):
    if not isinstance(response_json, dict):
        raise Exception("Gemini API 响应不是 JSON 对象")

    candidates = response_json.get("candidates", [])

    if not candidates:
        if "error" in response_json:
            err = response_json.get("error", {})

            if isinstance(err, dict):
                raise Exception(err.get("message", json.dumps(err, ensure_ascii=False)))

            raise Exception(str(err))

        raise Exception("API 响应中未包含 candidates")

    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            inline_data = part.get("inlineData")

            if inline_data and inline_data.get("data"):
                print("✓ 成功从 Gemini inlineData 中提取图片")
                return inline_data["data"]

            file_data = part.get("fileData")

            if file_data and file_data.get("fileUri"):
                image_url = file_data["fileUri"]
                print(f"从 Gemini fileData 中提取到图片 URL: {image_url}")

                img_response = requests.get(image_url, timeout=download_timeout, verify=False)

                if img_response.status_code == 200:
                    print("✓ 成功下载 Gemini fileData 图片 URL")
                    return base64.b64encode(img_response.content).decode("utf-8")

                raise Exception(
                    f"下载 Gemini fileData 图片失败，HTTP {img_response.status_code}: "
                    f"{img_response.text[:300]}"
                )

            text = part.get("text", "")

            if text:
                data_uri_pattern = r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)"
                data_uri_match = re.search(data_uri_pattern, text)

                if data_uri_match:
                    print("✓ 成功从 Gemini 文本 data URI 中提取图片")
                    return data_uri_match.group(2)

                markdown_url_pattern = r"!\[[^\]]*\]\((https?://[^\)]+)\)"
                markdown_matches = re.findall(markdown_url_pattern, text)

                if markdown_matches:
                    image_url = markdown_matches[0]
                    print(f"从 Markdown 中提取到图片 URL: {image_url}")

                    img_response = requests.get(image_url, timeout=download_timeout, verify=False)

                    if img_response.status_code == 200:
                        print("✓ 成功下载 Gemini Markdown 图片 URL")
                        return base64.b64encode(img_response.content).decode("utf-8")

                    raise Exception(f"下载图片失败，HTTP {img_response.status_code}")

                direct_url_pattern = r"https?://[^\s\"\']+\.(?:jpg|jpeg|png|webp|gif)(?:\?[^\s\"\']*)?"
                direct_url_matches = re.findall(direct_url_pattern, text)

                if direct_url_matches:
                    image_url = direct_url_matches[0]
                    print(f"从文本中提取到图片 URL: {image_url}")

                    img_response = requests.get(image_url, timeout=download_timeout, verify=False)

                    if img_response.status_code == 200:
                        print("✓ 成功下载 Gemini 文本图片 URL")
                        return base64.b64encode(img_response.content).decode("utf-8")

                    raise Exception(f"下载图片失败，HTTP {img_response.status_code}")

    raise Exception("Gemini API 响应中未找到图片数据")


def send_gemini_request(
    api_key,
    endpoint,
    model,
    prompt,
    reference_images,
    aspect_ratio="auto",
    image_size="1K",
    request_timeout=300,
    download_timeout=300,
    async_initial_delay=30,
    async_poll_interval=5,
    async_max_wait=1800,
    progress=None,
    session=None,
    request_id=None,
    is_cancelled=lambda: False,
    task_metadata=None,
):
    """Submit native Gemini JSON to the Gateway's durable async queue."""
    endpoint = _normalize_endpoint(endpoint)
    url = f"{endpoint}/v1beta/models/{model}:generateContent"
    parts = build_gemini_parts(prompt, reference_images)
    generation_config = {
        "responseModalities": ["IMAGE"],
    }
    image_config = {}
    if aspect_ratio and aspect_ratio != "auto":
        image_config["aspectRatio"] = aspect_ratio
    if image_size:
        image_config["imageSize"] = image_size
    if image_config:
        generation_config["imageConfig"] = image_config
    payload = {
        "async": True,
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": generation_config,
    }
    owns_session = session is None
    session = session or _new_http_session()
    request_id = request_id or f"yaliai_plugin_{int(time.time() * 1000)}_{threading.get_ident()}_{uuid.uuid4().hex}"
    print(f"发送 Gemini 请求到: {url}")
    print(f"模型: {model}")
    print(f"图像比例: {aspect_ratio}")
    print(f"图像大小: {image_size}")
    print(f"参考图数量: {len(_normalize_reference_images(reference_images))}")

    try:
        accepted = _submit_async_request(
            session,
            url,
            api_key,
            request_id,
            request_timeout,
            json_payload=payload,
            task_metadata=task_metadata,
        )
        return _poll_async_task(
            session,
            endpoint,
            api_key,
            accepted,
            request_timeout,
            async_initial_delay,
            async_poll_interval,
            async_max_wait,
            progress or (lambda *_: None),
            is_cancelled,
        )
    finally:
        if owns_session:
            session.close()


# ===================== OpenAI Images legacy code (unused) =====================

def build_gpt_image_size(aspect_ratio, image_size):
    """
    OpenAI Images 尺寸自动映射。

    前端只需要选择：
    - aspect_ratio: 1:1 / 16:9 / 9:16 / 4:3 / 3:4 / 3:2 / 2:3 / 21:9 / auto
    - image_size: 1K / 2K / 4K

    实际请求会转换成：
    - 1024x1024
    - 2048x1152
    - 3840x2160
    等接口要求的真实 size。
    """
    aspect_ratio = str(aspect_ratio or "1:1").strip()
    image_size = str(image_size or "1K").strip().upper()

    if not aspect_ratio:
        aspect_ratio = "1:1"

    if aspect_ratio == "auto":
        aspect_ratio = "1:1"

    if image_size not in GPT_IMAGE_SIZE_MAP:
        print(f"[OpenAI Images size] 不支持的 image_size={image_size}，回退到 1K")
        image_size = "1K"

    size_map = GPT_IMAGE_SIZE_MAP[image_size]

    if aspect_ratio not in size_map:
        raise Exception(
            f"OpenAI Images 不支持当前比例: {aspect_ratio}。"
            f"当前尺寸表仅支持: {', '.join(size_map.keys())}。"
            f"请改用支持的比例，或切换到 Gemini / 香蕉模型。"
        )

    size = size_map[aspect_ratio]

    print(
        f"[OpenAI Images size] "
        f"前端比例={aspect_ratio}, "
        f"前端档位={image_size}, "
        f"实际请求size={size}"
    )

    return size


def build_gpt_image_submit_model(image_size):
    """
    旧版 OpenAI Images 模型名映射，仅供旧代码路径参考。

    UI / 插件模型名仍然保持：
        OpenAI Images

    实际请求模型名：
        1K / 2K / 4K -> 旧版模型名映射
    """
    normalized_image_size = str(image_size or "1K").strip().upper()

    if normalized_image_size == "4K":
        return "gpt-image-2"

    return "gpt-image-2"


def extract_gpt_image_result(data):
    """
    从旧版 OpenAI Images 返回中提取图片。

    兼容以下返回格式：

    1. OpenAI 风格：
    {
        "data": [
            {
                "url": "https://..."
            }
        ]
    }

    2. OpenAI base64 风格：
    {
        "data": [
            {
                "b64_json": "..."
            }
        ]
    }

    3. 简化 url：
    {
        "url": "https://..."
    }

    4. 简化 b64：
    {
        "b64_json": "..."
    }
    """
    if not isinstance(data, dict):
        return {"type": "", "value": ""}

    if isinstance(data.get("data"), list) and data["data"]:
        item = data["data"][0]

        if isinstance(item, dict):
            if item.get("url"):
                return {"type": "url", "value": item["url"]}

            if item.get("b64_json"):
                return {"type": "b64", "value": item["b64_json"]}

    if data.get("url"):
        return {"type": "url", "value": data["url"]}

    if data.get("b64_json"):
        return {"type": "b64", "value": data["b64_json"]}

    return {"type": "", "value": ""}


def _parse_gpt_response_or_raise(response):
    """
    解析旧版 OpenAI Images 接口响应。
    """
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text[:1500]}")

    try:
        data = response.json()
    except Exception:
        raise Exception(f"OpenAI Images 接口返回非 JSON: {response.text[:1500]}")

    print("OpenAI Images 接口返回:")

    try:
        print(json.dumps(data, ensure_ascii=False)[:1500])
    except Exception:
        print(data)

    result = extract_gpt_image_result(data)

    if not result["value"]:
        raise Exception(
            "OpenAI Images 响应中未找到图片 url/b64_json: "
            + json.dumps(data, ensure_ascii=False)[:1200]
        )

    return result


def _legacy_send_gpt_image_request(
    api_key,
    endpoint,
    model,
    prompt,
    reference_images,
    aspect_ratio="1:1",
    image_size="1K",
    request_timeout=300,
    download_timeout=300,
):
    """
    旧版同步图片生成请求（当前入口不会调用）。

    规则：

    1. 接口路径保持 OpenAI 图片接口风格：
       - 文生图：/v1/images/generations
       - 图生图：/v1/images/edits

    2. UI 模型名保持：
       - OpenAI Images

    3. 实际请求模型名：
        - image_size -> 旧版模型映射

    4. 固定参数：
       - quality = medium
       - n = 1

    5. size 保持自动映射：
       - 1K / 2K / 4K + aspect_ratio 自动转换成真实分辨率。
    """
    endpoint = _normalize_endpoint(endpoint)

    # 自动映射真实 size，例如：
    # 1K + 1:1 -> 1024x1024
    # 2K + 16:9 -> 2048x1152
    # 4K + 16:9 -> 3840x2160
    size = build_gpt_image_size(aspect_ratio, image_size)

    # 根据 1K / 2K / 4K 决定实际请求模型名
    submit_model = build_gpt_image_submit_model(image_size)

    # 写死参数
    quality = "medium"
    n = 1

    ref_pack = _collect_valid_reference_images(reference_images)
    local_refs = ref_pack["local"]
    url_refs = ref_pack["urls"]

    print("发送旧版 OpenAI Images 请求")
    print(f"Endpoint: {endpoint}")
    print(f"UI模型: {model}")
    print(f"实际请求模型: {submit_model}")
    print(f"比例: {aspect_ratio}")
    print(f"档位: {image_size}")
    print(f"实际请求 size: {size}")
    print(f"quality: {quality}")
    print(f"n: {n}")
    print(f"本地参考图数量: {len(local_refs)}")
    print(f"URL参考图数量: {len(url_refs)}")

    # =========================================================
    # 1. 有本地参考图：图生图 edits multipart
    #
    # 等价 curl：
    #
    # curl {endpoint}/v1/images/edits \
    #   -H "Authorization: Bearer sk-xxx" \
    #   -F "model=gpt-image-2" \
    #   -F "prompt=..." \
    #   -F "size=1024x1024" \
    #   -F "quality=medium" \
    #   -F "n=1" \
    #   -F "image=@/path/to/source.png"
    #
    # 多垫图时使用重复 image 字段：
    #   -F "image=@source1.png"
    #   -F "image=@source2.png"
    # =========================================================
    if local_refs:
        url = f"{endpoint}/v1/images/edits"

        valid_paths = []

        for p in local_refs:
            if p and os.path.exists(p) and os.path.getsize(p) > 0:
                valid_paths.append(os.path.abspath(p))

        if not valid_paths:
            raise Exception("图生图需要垫图，但未找到有效本地图片")

        form_data = {
            "model": submit_model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": str(n),
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        file_handles = []
        files = []

        try:
            for img_path in valid_paths:
                fh = open(img_path, "rb")
                file_handles.append(fh)

                files.append(
                    (
                        "image",
                        (
                            os.path.basename(img_path),
                            fh,
                            guess_mime_type(img_path),
                        )
                    )
                )

                print(f"[OpenAI Images] 添加参考图字段 image: {img_path}")

            print("=" * 50)
            print(f"请求 URL: {url}")
            print("请求模式: 图生图 edits multipart")
            print("图片字段: image")
            print(f"图片数量: {len(files)}")
            print(f"data: {form_data}")
            print("=" * 50)

            response = requests.post(
                url,
                headers=headers,
                data=form_data,
                files=files,
                timeout=request_timeout,
            )

            return _parse_gpt_response_or_raise(response)

        finally:
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass

    # =========================================================
    # 2. 只有 URL 参考图：图生图 edits JSON
    #
    # 注意：
    # 这里保留原代码能力：
    # - 单 URL 使用 image_url
    # - 多 URL 使用 image_urls
    #
    # 如果服务商不支持 URL 垫图，可以只使用本地垫图。
    # =========================================================
    if url_refs:
        url = f"{endpoint}/v1/images/edits"

        payload = {
            "model": submit_model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n,
        }

        if len(url_refs) == 1:
            payload["image_url"] = url_refs[0]
        else:
            payload["image_urls"] = url_refs

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        print("=" * 50)
        print(f"请求 URL: {url}")
        print("请求模式: 图生图 edits JSON image_url/image_urls")
        print(f"payload: {json.dumps(payload, ensure_ascii=False)[:1500]}")
        print("=" * 50)

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=request_timeout,
        )

        return _parse_gpt_response_or_raise(response)

    # =========================================================
    # 3. 无参考图：文生图 generations JSON
    #
    # 等价 curl：
    #
    # curl {endpoint}/v1/images/generations \
    #   -H "Authorization: Bearer sk-xxx" \
    #   -H "Content-Type: application/json" \
    #   -d '{
    #     "model": "gpt-image-2",
    #     "prompt": "...",
    #     "size": "1024x1024",
    #     "quality": "medium",
    #     "n": 1
    #   }'
    # =========================================================
    url = f"{endpoint}/v1/images/generations"

    payload = {
        "model": submit_model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": n,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print("=" * 50)
    print(f"请求 URL: {url}")
    print("请求模式: 文生图 generations JSON")
    print(f"payload: {json.dumps(payload, ensure_ascii=False)[:1500]}")
    print("=" * 50)

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
    )

    return _parse_gpt_response_or_raise(response)

# ===================== 鸭梨 AI OpenAI Images 异步接口 =====================

def send_gpt_image_request(
    api_key,
    endpoint,
    model,
    prompt,
    reference_images,
    aspect_ratio="1:1",
    image_size="1K",
    quality="medium",
    request_timeout=300,
    download_timeout=300,
    async_initial_delay=30,
    async_poll_interval=5,
    async_max_wait=1800,
    progress=None,
    session=None,
    request_id=None,
    is_cancelled=lambda: False,
    task_metadata=None,
):
    """Submit OpenAI Images JSON/multipart input to the durable async API."""
    endpoint = _normalize_endpoint(endpoint)
    size = build_gpt_image_size(aspect_ratio, image_size)
    submission_metadata = dict(task_metadata or {})
    submission_metadata["request_size"] = size
    ref_pack = _collect_valid_reference_images(reference_images)
    local_refs = ref_pack["local"]
    url_refs = ref_pack["urls"]
    owns_session = session is None
    session = session or _new_http_session()
    request_id = request_id or f"yaliai_plugin_{int(time.time() * 1000)}_{threading.get_ident()}_{uuid.uuid4().hex}"

    common = {
        "model": model,
        "prompt": str(prompt),
        "size": size,
        "quality": quality,
        "n": 1,
        "response_format": "url",
        "async": True,
    }
    files = []
    handles = []
    try:
        if local_refs:
            url = f"{endpoint}/v1/images/edits"
            for image_path in local_refs:
                handle = open(image_path, "rb")
                handles.append(handle)
                files.append(("image", (os.path.basename(image_path), handle, guess_mime_type(image_path))))
            accepted = _submit_async_request(
                session,
                url,
                api_key,
                request_id,
                request_timeout,
                form_data=common,
                files=files,
                task_metadata=submission_metadata,
            )
        elif url_refs:
            url = f"{endpoint}/v1/images/edits"
            payload = dict(common)
            payload["image"] = url_refs[0] if len(url_refs) == 1 else url_refs
            accepted = _submit_async_request(
                session,
                url,
                api_key,
                request_id,
                request_timeout,
                json_payload=payload,
                task_metadata=submission_metadata,
            )
        else:
            url = f"{endpoint}/v1/images/generations"
            accepted = _submit_async_request(
                session,
                url,
                api_key,
                request_id,
                request_timeout,
                json_payload=common,
                task_metadata=submission_metadata,
            )
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
        if owns_session and "accepted" not in locals():
            session.close()

    try:
        return _poll_async_task(
            session,
            endpoint,
            api_key,
            accepted,
            request_timeout,
            async_initial_delay,
            async_poll_interval,
            async_max_wait,
            progress or (lambda *_: None),
            is_cancelled,
        )
    finally:
        if owns_session:
            session.close()


# ===================== Legacy synchronous entry (unused) =====================

def _legacy_generate(context):
    print("\n" + "=" * 60)
    print("旧版同步入口未启用")
    print("=" * 60)

    p = _merge_runtime_params(context)

    prompt = context.get("prompt", "")
    reference_images = context.get("reference_images", {})
    output_dir = _get_output_dir(context)

    api_key = str(p.get("api_key", "") or "").strip()
    endpoint = _normalize_endpoint(p.get("endpoint", "https://api.yaliai.com"))
    model = _normalize_model(p.get("model", "gemini-3.1-flash-image-preview"))
    aspect_ratio = str(p.get("aspect_ratio", "auto") or "auto").strip()
    image_size = str(p.get("image_size", "1K") or "1K").strip()
    request_timeout = _safe_int(p.get("request_timeout", 300), 300)
    download_timeout = _safe_int(p.get("download_timeout", 300), 300)
    retry_count = _safe_int(p.get("retry_count", 0), 0)

    config_snapshot = _load_config()

    print("\n===== 参数来源调试 =====")
    print(f"config.json 路径: {_CONFIG_PATH}")
    print(f"config.json 存在: {_CONFIG_PATH.exists()}")
    print(f"config.json.model: {config_snapshot.get('model', '【空】')}")
    print(f"最终 model: {model}")
    print("======================\n")

    print("===== 生成参数 =====")
    print(f"提示词: {prompt}")
    print(f"参考图片数量: {len(_normalize_reference_images(reference_images))}")
    print(f"API Key: {'已设置 (' + str(len(api_key)) + ' 字符)' if api_key else '未设置'}")
    print(f"Endpoint: {endpoint}")
    print(f"模型: {model}")
    print(f"图像比例: {aspect_ratio}")
    print(f"图像大小: {image_size}")
    print(f"请求超时: {request_timeout} 秒")
    print(f"下载超时: {download_timeout} 秒")
    print(f"旧版重试次数: {retry_count}")
    print("==================\n")

    if not api_key:
        raise Exception("PLUGIN_ERROR:::未设置 API Key，请在插件设置中填写")

    if not endpoint:
        raise Exception("PLUGIN_ERROR:::未设置 Endpoint")

    if not prompt or not str(prompt).strip():
        raise Exception("PLUGIN_ERROR:::提示词不能为空")

    if not output_dir:
        raise Exception("PLUGIN_ERROR:::未提供输出目录 output_dir/project_path")

    os.makedirs(output_dir, exist_ok=True)

    progress_callback = context.get("progress_callback")

    def progress(text, percent=None):
        if not progress_callback:
            return

        try:
            if percent is None:
                progress_callback(text)
            else:
                progress_callback(text, percent)

        except TypeError:
            try:
                progress_callback(text)
            except Exception:
                pass

        except Exception:
            pass

    generated_files = []
    max_attempts = max(0, retry_count) + 1

    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                print(f"\n第 {attempt + 1}/{max_attempts} 次尝试...")
                time.sleep(2)
            else:
                print("正在调用 OpenAI Images API...")

            progress("生成中", 10)

            if model in GPT_IMAGE_MODELS:
                result = send_gpt_image_request(
                    api_key=api_key,
                    endpoint=endpoint,
                    model=model,
                    prompt=prompt,
                    reference_images=reference_images,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    request_timeout=request_timeout,
                    download_timeout=download_timeout,
                )

                progress("生成中", 80)

                if result["type"] == "url":
                    output_path = download_url_to_output(
                        result["value"],
                        context,
                        output_dir,
                        download_timeout=download_timeout,
                    )

                elif result["type"] == "b64":
                    output_path = save_image_base64_to_output(
                        result["value"],
                        context,
                        output_dir,
                    )

                else:
                    raise Exception("未知图片返回类型")

            else:
                if model not in GEMINI_MODELS:
                    print(f"⚠️ 当前模型不在预设 Gemini 列表中: {model}，将继续按 Gemini 格式请求")

                # Gemini / 香蕉模型保持原逻辑
                image_base64 = send_gemini_request(
                    api_key=api_key,
                    endpoint=endpoint,
                    model=model,
                    prompt=prompt,
                    reference_images=reference_images,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    request_timeout=request_timeout,
                    download_timeout=download_timeout,
                )

                progress("生成中", 80)

                output_path = save_image_base64_to_output(
                    image_base64,
                    context,
                    output_dir,
                )

            generated_files.append(output_path)

            progress("生成中", 100)

            print(f"✓ 成功生成图片: {output_path}")
            break

        except Exception as e:
            is_last_attempt = attempt == max_attempts - 1

            if is_last_attempt:
                error_msg = f"生成失败（已尝试 {max_attempts} 次）: {str(e)}"
                print(f"❌ {error_msg}")

                import traceback
                traceback.print_exc()

                raise Exception(f"PLUGIN_ERROR:::{error_msg}")

            print(f"⚠️ 第 {attempt + 1} 次尝试失败: {str(e)}")
            print("将在 2 秒后重试...")
            time.sleep(2)

    print("\n" + "=" * 60)
    print(f"旧版同步入口完成，共生成 {len(generated_files)} 张图片")
    print("=" * 60 + "\n")

    return generated_files


# ===================== 鸭梨 AI 异步生成入口 =====================

def generate(context):
    """Generate one host task, or a host-provided batch, in deterministic order."""
    context = context or {}
    params = _merge_runtime_params(context)
    prompt = str(context.get("prompt", "") or "").strip()
    reference_images = context.get("reference_images", {}) or {}
    output_dir = _get_output_dir(context)
    endpoint = _normalize_endpoint(params.get("endpoint", "https://api.yaliai.com"))
    model = _normalize_model(params.get("model", "gemini-3.1-flash-image-preview"))
    api_key = _api_key_for_model(params, model)
    aspect_ratio = str(params.get("aspect_ratio", "16:9") or "16:9").strip()
    image_size = str(params.get("image_size", "4K") or "4K").strip().upper()
    quality = str(params.get("quality", "medium") or "medium").strip().lower()
    request_timeout = _GATEWAY_HTTP_TIMEOUT_SECONDS
    download_timeout = _IMAGE_DOWNLOAD_TIMEOUT_SECONDS
    initial_delay = _ASYNC_INITIAL_DELAY_SECONDS
    poll_interval = _ASYNC_POLL_INTERVAL_SECONDS
    max_wait = _ASYNC_MAX_WAIT_SECONDS
    batch_num = _safe_int(context.get("batch_num", 1), 1)
    if batch_num < 1 or batch_num > 16:
        raise Exception("PLUGIN_ERROR:::batch_num 必须在 1 到 16 之间")
    output_positions = context.get("output_position")
    if not isinstance(output_positions, (list, tuple)):
        output_positions = []

    if not api_key:
        credential_label = "GPT-image-2 API Key" if model in GPT_IMAGE_MODELS else "Gemini API Key"
        raise Exception(f"PLUGIN_ERROR:::未设置 {credential_label}")
    if not endpoint:
        raise Exception("PLUGIN_ERROR:::未设置鸭梨 AI Gateway 地址")
    if not prompt:
        raise Exception("PLUGIN_ERROR:::提示词不能为空")
    if not output_dir:
        raise Exception("PLUGIN_ERROR:::未提供输出目录 output_dir/project_path")
    if quality not in {"low", "medium", "high"}:
        raise Exception("PLUGIN_ERROR:::画质必须是 low、medium 或 high")

    os.makedirs(output_dir, exist_ok=True)
    progress_callback = context.get("progress_callback")

    def progress(text, percent=None):
        if not progress_callback:
            return
        try:
            if percent is None:
                progress_callback(text)
            else:
                progress_callback(text, percent)
        except TypeError:
            try:
                progress_callback(text)
            except Exception:
                pass
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("鸭梨 AI 图像生成插件开始异步任务")
    print("=" * 60)
    print(f"模型: {model}; 比例: {aspect_ratio}; 档位: {image_size}; 画质: {quality}; 批次: {batch_num}")
    print(f"参考图数量: {len(_normalize_reference_images(reference_images))}")

    def is_cancelled():
        return _is_cancelled(context)

    session = _new_http_session()
    generated_files = []
    try:
        # The host normally expands a batch into independent tasks. This loop
        # is the safe fallback for direct plugin calls that still carry batch_num.
        for index in range(batch_num):
            position = output_positions[index] if index < len(output_positions) else index
            request_id = _new_request_id(context, position)
            task_metadata = {
                "model": model,
                "quality": quality,
                "image_size": image_size,
                "aspect_ratio": aspect_ratio,
                "protocol": "openai_image" if model in GPT_IMAGE_MODELS else "gemini",
                "viewer_index": _safe_int(context.get("viewer_index", 0), 0),
                "unique_name": str(context.get("unique_name", "") or ""),
                "generation_round": _safe_int(context.get("generation_round", 0), 0),
                "output_position": position,
                "batch_index": index,
                "batch_num": batch_num,
                "reference_image_count": len(_normalize_reference_images(reference_images)),
                "prompt_preview": _prompt_preview(prompt),
            }
            progress(f"正在生成第 {index + 1}/{batch_num} 张图片", 10 + int(index * 70 / batch_num))
            if is_cancelled():
                raise Exception("任务已被宿主取消")

            if model in GPT_IMAGE_MODELS:
                outputs = send_gpt_image_request(
                    api_key=api_key,
                    endpoint=endpoint,
                    model=model,
                    prompt=prompt,
                    reference_images=reference_images,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    quality=quality,
                    request_timeout=request_timeout,
                    download_timeout=download_timeout,
                    async_initial_delay=initial_delay,
                    async_poll_interval=poll_interval,
                    async_max_wait=max_wait,
                    progress=progress,
                    session=session,
                    request_id=request_id,
                    is_cancelled=is_cancelled,
                    task_metadata=task_metadata,
                )
            else:
                outputs = send_gemini_request(
                    api_key=api_key,
                    endpoint=endpoint,
                    model=model,
                    prompt=prompt,
                    reference_images=reference_images,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    request_timeout=request_timeout,
                    download_timeout=download_timeout,
                    async_initial_delay=initial_delay,
                    async_poll_interval=poll_interval,
                    async_max_wait=max_wait,
                    progress=progress,
                    session=session,
                    request_id=request_id,
                    is_cancelled=is_cancelled,
                    task_metadata=task_metadata,
                )

            if not outputs:
                raise Exception(f"第 {index + 1} 个鸭梨 AI 任务完成但没有图片输出")
            if len(outputs) > 1:
                print(f"警告：任务返回 {len(outputs)} 张图片；当前宿主槽位只接收第一张")
            output = outputs[0]
            task_id = str(output.get("task_id", "") or "")
            progress(f"正在下载第 {index + 1}/{batch_num} 张图片", 82 + int(index * 12 / batch_num))
            try:
                if output.get("type") == "url":
                    image_url = _absolute_gateway_url(endpoint, output.get("value"))
                    if not image_url:
                        raise Exception("任务结果缺少图片 URL")
                    path = download_url_to_output(
                        image_url,
                        context,
                        output_dir,
                        download_timeout=download_timeout,
                        position_override=position,
                        session=session,
                        is_cancelled=is_cancelled,
                    )
                elif output.get("type") == "b64":
                    image_url = ""
                    path = save_image_base64_to_output(output.get("value"), context, output_dir, position)
                else:
                    raise Exception("任务结果包含未知图片格式")
            except Exception as delivery_error:
                _record_async_task(
                    "delivery_failed",
                    task_id=task_id,
                    status="download_failed",
                    error=str(delivery_error),
                    **task_metadata,
                )
                raise
            generated_files.append(path)
            _record_async_task(
                "delivered",
                task_id=task_id,
                status="success",
                output_path=os.path.abspath(path),
                output_url=image_url,
                output_type=output.get("type", ""),
                **task_metadata,
            )
            progress(f"第 {index + 1}/{batch_num} 张图片已保存", 86 + int((index + 1) * 14 / batch_num))

        print(f"鸭梨 AI 异步任务完成，共保存 {len(generated_files)} 张图片")
        return generated_files
    except Exception as exc:
        print(f"鸭梨 AI 异步任务失败: {exc}")
        raise Exception(f"PLUGIN_ERROR:::{exc}") from exc
    finally:
        session.close()


# ===================== 初始化 =====================

_ensure_config_exists()

print("[Yali AI Image] 插件已加载")
print(f"[Yali AI Image] 配置文件: {_CONFIG_PATH}")
print(f"[Yali AI Image] 配置文件存在: {_CONFIG_PATH.exists()}")

try:
    _init_config = _load_config()
    print(f"[Yali AI Image] 当前配置 model = {_init_config.get('model', '【空】')}")
except Exception:
    pass


# ===================== 本地测试入口 =====================

if __name__ == "__main__":
    context = {
        "prompt": "Turn this image into a flat app icon style",
        "reference_images": {},
        "output_dir": "./",
        "viewer_index": 1,
        "unique_name": "testabc",
        "generation_round": 0,
        "output_position": [0],
        "plugin_params": {
            "endpoint": "https://api.yaliai.com",
            "model": "gpt-image-2",
            "aspect_ratio": "16:9",
            "image_size": "4K",
            "quality": "medium",
            "gpt_api_key": "sk-xxxx",
        },
    }

    print(generate(context))
