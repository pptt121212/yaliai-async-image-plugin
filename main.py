# -*- coding: utf-8 -*-
"""
鸭梨 AI 图像生成插件

参数同步：
- 在字字动画中运行时，配置和插件状态保存在 user_resources/plugins/
- 前端通过 PluginSDK.saveParam() 持久化普通参数
- generate() 合并默认值、运行配置和当前宿主调用参数

支持：
- Yali AI Gateway native Gemini and OpenAI Images endpoints
- Durable async submission, task polling, and local image delivery
- OpenAI Images reference files use multipart uploads; Gemini uses inlineData

"""

import os
import re
import base64
import hashlib
import shutil
import time
import json
import threading
import tempfile
import uuid
import requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from requests.adapters import HTTPAdapter

try:
    from urllib3.exceptions import InsecureRequestWarning

    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
except Exception:
    pass

from pathlib import Path
from io import BytesIO
from PIL import Image, ImageOps
from urllib.parse import urljoin, urlparse


_PLUGIN_FILE = __file__
plugin_dir = Path(__file__).parent


# ===================== 插件状态目录 =====================

def _resolve_state_dir():
    """Keep user data outside the installed plugin directory when hosted.

    A distributed plugin lives below ``_internal/plugins``. The host reserves
    its sibling ``user_resources/plugins`` tree for mutable user data, which
    survives plugin upgrades and is also the source for ``plugin_params``.
    Standalone tests retain the plugin-local fallback.
    """
    try:
        internal_dir = plugin_dir.parents[2]
        if internal_dir.name == "_internal":
            app_dir = internal_dir.parent
            user_resources = app_dir / "user_resources"
            if user_resources.is_dir():
                return user_resources / "plugins" / plugin_dir.parent.name / plugin_dir.name
    except IndexError:
        pass
    return plugin_dir


_LEGACY_STATE_DIR = plugin_dir
_STATE_DIR = _resolve_state_dir()
_CONFIG_PATH = _STATE_DIR / "config.json"
_CONFIGURED_REFERENCE_DIR = _STATE_DIR / "configured_references"
_config_lock = threading.Lock()
_ASYNC_TASK_LOG_PATH = _STATE_DIR / "async_tasks.jsonl"
_async_task_log_lock = threading.Lock()
_TASK_THUMBNAIL_DIR = _STATE_DIR / "task_thumbnails"
_manual_upscale_jobs_lock = threading.Lock()
_manual_upscale_jobs = set()
_GATEWAY_ENDPOINT = "https://api.yaliai.com"
_HOST_TOOL_CALL_ENDPOINT = "http://127.0.0.1:8766/v1/tools/call"
_HOST_TOOL_CALL_TIMEOUT_SECONDS = 12
_OPENAI_IMAGE_OUTPUT_FORMAT = "jpeg"
_DEFAULT_UPSCALE_PROMPT = (
    "现在对这张图进行全景像素超分（Panorama Super-Resolution）与重绘。"
    "请将图像精细度和文字边缘细节提升至 {image_size} 电影级分辨率。"
    "场景清晰不允许存在锯齿和噪点，颜色纯净。请在画面中原地追加细节。"
    "保持图像 {aspect_ratio} 比例，不要因为限制图像比例而使用变形的素材和文字。"
    "图像比例不对将判定为任务失败！"
)


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


def _migrate_legacy_state():
    """Move pre-user_resources state once without replacing current data."""
    if _STATE_DIR == _LEGACY_STATE_DIR:
        return
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "async_tasks.jsonl"):
            legacy = _LEGACY_STATE_DIR / name
            target = _STATE_DIR / name
            if legacy.is_file() and not target.exists():
                shutil.copy2(legacy, target)
        for name in ("configured_references", "task_thumbnails"):
            legacy = _LEGACY_STATE_DIR / name
            target = _STATE_DIR / name
            if legacy.is_dir() and not target.exists():
                shutil.copytree(legacy, target)
    except Exception as error:
        print(f"[Yali AI Image] 迁移旧插件状态失败: {error}")


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

            tmp_path = _CONFIG_PATH.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, _CONFIG_PATH)

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
    "endpoint": _GATEWAY_ENDPOINT,
    "gpt_api_key": "",
    "gemini_api_key": "",
    "model": "gemini-3.1-flash-image-preview",
    "aspect_ratio": "16:9",
    "image_size": "4K",
    "quality": "medium",
    "generation_mode": "default",
    "upscale_model": "gemini-3-pro-image-preview",
    "upscale_image_size": "4K",
    "upscale_prompt": _DEFAULT_UPSCALE_PROMPT,
    "local_result_max_mb": 5,
    "configured_reference_images": [],
}

# These are execution guarantees, not end-user tuning knobs. A gateway image
# request can legitimately take minutes, while polling is cheap and cancellable.
_GATEWAY_HTTP_TIMEOUT_SECONDS = 300
_IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 300
_ASYNC_INITIAL_DELAY_SECONDS = 20
_ASYNC_POLL_INTERVAL_SECONDS = 2
_ASYNC_MAX_WAIT_SECONDS = 1800
_MAX_BATCH_NUM = 100
_LOCAL_MAX_ACTIVE_TASKS = 40
_LOCAL_MAX_REFERENCE_TASKS = 20
_LOCAL_MAX_DELIVERY_TASKS = 8
_RETIRED_PARAM_KEYS = {
    "api_key",
    "request_timeout",
    "download_timeout",
    "async_initial_delay",
    "async_poll_interval",
    "async_max_wait",
    "retry_count",
    "upscale_trigger_size",
}

_REFERENCE_IMAGE_MAX_LONG_EDGE = 4096
_REFERENCE_IMAGE_SOFT_PIXELS = 12 * 1000 * 1000
_REFERENCE_IMAGE_MAX_PIXELS = 16 * 1000 * 1000
_REFERENCE_IMAGE_MIN_LONG_EDGE = 320
_REFERENCE_IMAGE_THRESHOLD_BYTES = 2 * 1024 * 1024
_REFERENCE_IMAGE_TARGET_BYTES = _REFERENCE_IMAGE_THRESHOLD_BYTES - 1
_REFERENCE_IMAGE_MIN_TARGET_BYTES = 300 * 1024
_REFERENCE_IMAGE_TARGET_BYTES_PER_PIXEL = 1.2
_MANUAL_UPSCALE_BATCH_LIMIT = 40
_SUPPORTED_ASPECT_RATIOS = {
    "1:1": 1.0,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "21:9": 21 / 9,
}


class _FifoGate:
    """A cancellable FIFO concurrency gate shared by every plugin call."""

    def __init__(self, limit):
        self._limit = max(1, int(limit))
        self._active = 0
        self._waiters = deque()
        self._condition = threading.Condition()

    def acquire(self, is_cancelled):
        token = object()
        with self._condition:
            self._waiters.append(token)
            while self._active >= self._limit or self._waiters[0] is not token:
                if is_cancelled():
                    self._waiters.remove(token)
                    self._condition.notify_all()
                    raise Exception("任务已被宿主取消")
                self._condition.wait(timeout=0.5)
            self._waiters.popleft()
            self._active += 1

    def release(self):
        with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify_all()


# Keep a local workstation responsive when the host queues many frames. A
# reference image consumes upload bandwidth as well as an active task slot.
_local_task_gate = _FifoGate(_LOCAL_MAX_ACTIVE_TASKS)
_local_reference_gate = _FifoGate(_LOCAL_MAX_REFERENCE_TASKS)
_local_delivery_gate = _FifoGate(_LOCAL_MAX_DELIVERY_TASKS)
_local_reference_compression_gate = _FifoGate(1)


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
    if (
        str(current.get("generation_mode", "default") or "default").strip().lower() == "default"
        and _normalize_model(current.get("upscale_model")) == "gpt-image-2"
    ):
        migration["upscale_model"] = "gemini-3-pro-image-preview"
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

_IMAGE_SIZE_RANKS = {size: index for index, size in enumerate(IMAGE_SIZES, start=1)}


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

    return data


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _normalize_image_size(value, default="1K"):
    size = str(value or default).strip().upper()
    return size if size in _IMAGE_SIZE_RANKS else default


def _prompt_preview(value, limit=120):
    """Keep only a bounded, Unicode-safe prompt hint for local task logs."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _render_upscale_prompt(template, image_size, aspect_ratio):
    text = str(template or _DEFAULT_UPSCALE_PROMPT).strip()
    if not text:
        raise Exception("PLUGIN_ERROR:::超分提示词不能为空")
    replacements = {
        "{{image_size}}": str(image_size),
        "{image_size}": str(image_size),
        "{{aspect_ratio}}": str(aspect_ratio),
        "{aspect_ratio}": str(aspect_ratio),
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def _normalize_endpoint(endpoint):
    value = str(endpoint or _GATEWAY_ENDPOINT).strip()
    if not value:
        value = _GATEWAY_ENDPOINT

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise Exception("PLUGIN_ERROR:::API URL 必须是有效的 http 或 https 地址")
    if parsed.query or parsed.fragment:
        raise Exception("PLUGIN_ERROR:::API URL 不能包含查询参数或片段")

    return value.rstrip("/")


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


def _configured_reference_paths():
    """Read only the plugin-owned reference configuration, never host params."""
    config = _load_config()
    values = config.get("configured_reference_images", []) if isinstance(config, dict) else []
    if not isinstance(values, (list, tuple)):
        return []

    root = _CONFIGURED_REFERENCE_DIR.resolve()
    paths = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and resolved.stat().st_size > 0:
            paths.append(str(resolved))
    return paths[:8]


def _merge_reference_images(host_references, configured_paths):
    """Keep host order, then append the plugin's explicitly configured images."""
    values = []
    if host_references:
        values.append(host_references)
    values.extend(configured_paths or [])
    return _normalize_reference_images(values)


def _configured_reference_info():
    result = []
    for path in _configured_reference_paths():
        try:
            item = Path(path)
            result.append({"name": item.name, "path": path, "size": item.stat().st_size})
        except OSError:
            continue
    return result


def _remove_configured_reference_files(paths):
    root = _CONFIGURED_REFERENCE_DIR.resolve()
    for value in paths or []:
        try:
            path = Path(value).expanduser().resolve()
            path.relative_to(root)
            if path.is_file():
                path.unlink()
        except (OSError, ValueError):
            continue


def _save_configured_references(images):
    """Persist uploaded references without modifying the user's source files."""
    if not isinstance(images, list) or len(images) > 8:
        return {"ok": False, "error": "参考图最多 8 张"}

    _CONFIGURED_REFERENCE_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(
        prefix="configured-references-", dir=str(_CONFIGURED_REFERENCE_DIR.parent)
    ))
    saved = []
    try:
        _CONFIGURED_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(images):
            if not isinstance(item, dict):
                raise ValueError("参考图数据格式无效")
            data_url = str(item.get("data_url", "") or "")
            if "," not in data_url or not data_url.lower().startswith("data:"):
                raise ValueError("参考图必须是本地上传文件")
            header, encoded = data_url.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
            if not raw:
                raise ValueError("参考图为空文件")
            source_path = staging_dir / f"source_{index}.bin"
            source_path.write_bytes(raw)
            with Image.open(source_path) as opened:
                opened.verify()

            suffix = ".png"
            if "jpeg" in header.lower() or "jpg" in header.lower():
                suffix = ".jpg"
            elif "webp" in header.lower():
                suffix = ".webp"
            final_source = source_path
            if source_path.stat().st_size > _REFERENCE_IMAGE_THRESHOLD_BYTES:
                compressed = _compress_reference_file(
                    str(source_path), str(staging_dir), _REFERENCE_IMAGE_TARGET_BYTES
                )
                final_source = Path(compressed)
                suffix = ".jpg"

            target = staging_dir / f"reference_{index}{suffix}"
            if final_source != target:
                final_source.replace(target)
            saved.append(target)

        # Remove the previous managed files before installing the new set. The
        # target names are deterministic, so deleting old paths after the move
        # could delete the newly saved file when its name is unchanged.
        for old in _CONFIGURED_REFERENCE_DIR.glob("*"):
            if old.is_file():
                old.unlink()
        for source in saved:
            shutil.move(str(source), str(_CONFIGURED_REFERENCE_DIR / source.name))
        paths = [str((_CONFIGURED_REFERENCE_DIR / source.name).resolve()) for source in saved]
        if not _save_config({"configured_reference_images": paths}):
            raise RuntimeError("参考图配置保存失败")
        return {"ok": True, "images": _configured_reference_info()}
    except Exception as exc:
        return {"ok": False, "error": f"参考图保存失败: {exc}"}
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _normalize_reference_images(reference_images):
    """Flatten the host's reference-image shapes into an ordered path map.

    The editor normally sends ``{index: path}``, but preset images can also
    arrive as a list, a nested reference map, or objects containing ``path``
    / ``image_path``. Keeping this normalization at the boundary prevents a
    dict from being stringified and silently omitted by every protocol adapter.
    """
    if not reference_images:
        return {}

    path_keys = (
        "path",
        "file_path",
        "image_path",
        "local_path",
        "source_path",
        "url",
        "uri",
    )
    def scalar_path(value):
        if isinstance(value, (str, os.PathLike)):
            text = os.fspath(value).strip()
            return text or None
        if isinstance(value, dict):
            for key in path_keys:
                candidate = value.get(key)
                if isinstance(candidate, (str, os.PathLike)):
                    text = os.fspath(candidate).strip()
                    if text:
                        return text
        return None

    values = []

    def ordered_mapping_values(mapping):
        indexed = list(enumerate(mapping.items()))

        def sort_key(item):
            order, (key, _) = item
            text = str(key).strip()
            if re.fullmatch(r"\d+", text):
                return (0, int(text), order)
            return (1, order, order)

        return [value for _, (_, value) in sorted(indexed, key=sort_key)]

    def append(value):
        path = scalar_path(value)
        if path:
            values.append(path)
            return
        if isinstance(value, dict):
            for child in ordered_mapping_values(value):
                append(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                append(child)

    if isinstance(reference_images, dict):
        # Keep the host's top-level insertion order. A preset may interleave
        # first/last frames with a numbered reference MAP; moving the MAP to
        # the front changes the meaning of reference-image order.
        for value in reference_images.values():
            append(value)
    else:
        append(reference_images)

    result = {}
    seen = set()
    for path in values:
        if path.lower().startswith(("http://", "https://")):
            dedupe_key = path
        else:
            dedupe_key = os.path.normcase(os.path.abspath(path))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result[len(result)] = path
    return result


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


def _reference_target_bytes(params):
    del params
    return _REFERENCE_IMAGE_TARGET_BYTES


def _reference_target_bytes_for_pixels(max_target_bytes, pixels):
    density_target = int(max(1, pixels) * _REFERENCE_IMAGE_TARGET_BYTES_PER_PIXEL)
    return min(
        max(1, int(max_target_bytes)),
        max(_REFERENCE_IMAGE_MIN_TARGET_BYTES, density_target),
    )


def _encode_reference_jpeg(image, quality):
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", image.size, "white")
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")
        flattened.paste(rgb, mask=alpha)
        rgb.close()
        alpha.close()
        rgba.close()
        image = flattened
    else:
        image = image.convert("RGB")

    output = BytesIO()
    image.save(output, "JPEG", quality=quality, optimize=True, progressive=True)
    return output.getvalue()


def _compress_reference_file(
    path,
    staging_dir,
    max_target_bytes,
    is_cancelled=lambda: False,
    compress_threshold_bytes=None,
):
    """Create a bounded temporary JPEG without modifying the user's file."""
    path = os.path.abspath(str(path))
    source_size = os.path.getsize(path)
    if source_size <= 0:
        raise Exception(f"参考图为空文件: {os.path.basename(path)}")
    threshold = _REFERENCE_IMAGE_THRESHOLD_BYTES if compress_threshold_bytes is None else max(1, int(compress_threshold_bytes))
    if source_size <= threshold:
        return path

    with Image.open(path) as opened:
        width, height = opened.size
        # Let decoders that support native downsampling (notably JPEG) avoid
        # materializing the full source raster before Pillow resizes it.
        if width * height > _REFERENCE_IMAGE_SOFT_PIXELS and hasattr(opened, "draft"):
            opened.draft("RGB", (_REFERENCE_IMAGE_MAX_LONG_EDGE, _REFERENCE_IMAGE_MAX_LONG_EDGE))
        if is_cancelled():
            raise Exception("任务已被宿主取消；参考图压缩已停止")
        image = ImageOps.exif_transpose(opened)
        image.load()
        width, height = image.size
        if width < 1 or height < 1:
            raise Exception(f"参考图尺寸无效: {os.path.basename(path)}")

        target_bytes = _reference_target_bytes_for_pixels(
            max_target_bytes, width * height
        )

        needs_resize = (
            max(width, height) > _REFERENCE_IMAGE_MAX_LONG_EDGE
            or width * height > _REFERENCE_IMAGE_SOFT_PIXELS
        )
        current_width, current_height = width, height
        if needs_resize:
            scale = min(1.0, _REFERENCE_IMAGE_MAX_LONG_EDGE / max(width, height))
            if width * height > _REFERENCE_IMAGE_MAX_PIXELS:
                scale = min(scale, (_REFERENCE_IMAGE_MAX_PIXELS / (width * height)) ** 0.5)
            current_width = max(1, int(round(width * scale)))
            current_height = max(1, int(round(height * scale)))

        best = None
        # Use a small quality ladder, then estimate the next scale from the
        # actual encoded size. This avoids dozens of full JPEG encodes while
        # retaining a final size correction pass.
        bytes_per_pixel = source_size / max(1, width * height)
        if bytes_per_pixel >= 2.5:
            qualities = (92, 90, 88, 86, 84)
        elif source_size >= 5 * 1024 * 1024:
            qualities = (88, 86, 84, 82, 80)
        else:
            qualities = (90, 88, 86, 84, 82)
        for _ in range(6):
            if is_cancelled():
                raise Exception("任务已被宿主取消；参考图压缩已停止")
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            candidate_image = image.resize(
                (current_width, current_height), resampling
            ) if (current_width, current_height) != (width, height) else image.copy()
            current_best = None
            try:
                for quality in qualities:
                    candidate = _encode_reference_jpeg(candidate_image, quality)
                    if current_best is None or len(candidate) < len(current_best):
                        current_best = candidate
                    if best is None or len(candidate) < len(best):
                        best = candidate
                    if len(candidate) <= target_bytes:
                        break
            finally:
                candidate_image.close()
            if current_best is not None and len(current_best) <= target_bytes:
                break
            if max(current_width, current_height) <= _REFERENCE_IMAGE_MIN_LONG_EDGE:
                break
            size_ratio = (
                (target_bytes / max(1, len(current_best))) ** 0.5 * 0.96
                if current_best
                else 0.72
            )
            size_ratio = max(0.55, min(0.90, size_ratio))
            current_width = max(1, int(current_width * size_ratio))
            current_height = max(1, int(current_height * size_ratio))

    if not best or len(best) > target_bytes:
        raise Exception(f"参考图压缩失败: {os.path.basename(path)}")

    output_path = os.path.join(
        staging_dir,
        f"reference_{uuid.uuid4().hex}.jpg",
    )
    with open(output_path, "wb") as output:
        output.write(best)
    return output_path


def _local_result_target_bytes(params):
    try:
        megabytes = float(params.get("local_result_max_mb", _default_params["local_result_max_mb"]))
    except (TypeError, ValueError):
        megabytes = float(_default_params["local_result_max_mb"])
    megabytes = min(50.0, max(1.0, megabytes))
    return int(megabytes * 1024 * 1024)


def _compress_delivered_output(
    path,
    params,
    is_cancelled=lambda: False,
    allow_format_change=False,
):
    """Compress a fresh host result; existing replacements keep their format."""
    source = Path(path)
    target_bytes = _local_result_target_bytes(params)
    try:
        if not source.is_file() or source.stat().st_size <= target_bytes:
            return str(source)
        _local_reference_compression_gate.acquire(is_cancelled)
        try:
            with tempfile.TemporaryDirectory(prefix="yaliai-result-compress-", dir=str(source.parent)) as staging_dir:
                compressed_path = _compress_delivered_image_file(
                    str(source),
                    staging_dir,
                    target_bytes,
                    is_cancelled=is_cancelled,
                    allow_format_change=allow_format_change,
                )
                if allow_format_change and source.suffix.lower() not in {".jpg", ".jpeg"}:
                    replacement = source.with_suffix(".jpg")
                    os.replace(compressed_path, replacement)
                    source.unlink()
                    print(
                        f"[Yali AI Image] 新结果已转 JPEG 并交给宿主: {replacement.name} "
                        f"({replacement.stat().st_size / 1024 / 1024:.2f} MB)"
                    )
                    return str(replacement)
                if os.path.abspath(compressed_path) == os.path.abspath(str(source)):
                    return str(source)
                # Keep the host-visible path stable. The host may already hold
                # this exact path for a storyboard, character, or scene image.
                os.replace(compressed_path, source)
                print(
                    f"[Yali AI Image] 本地结果已压缩并替换: {source.name} "
                    f"({source.stat().st_size / 1024 / 1024:.2f} MB)"
                )
                return str(source)
        finally:
            _local_reference_compression_gate.release()
    except Exception as exc:
        # Image generation and host delivery have succeeded. A local size
        # optimization must never discard that successful result.
        print(f"[Yali AI Image] 本地结果压缩跳过: {exc}")
        return str(source)


def _encode_delivered_image(image, image_format, quality):
    """Encode a result without changing its file format."""
    output = BytesIO()
    normalized_format = str(image_format or "").upper()
    if normalized_format == "PNG":
        image.save(output, "PNG", optimize=True, compress_level=9)
    elif normalized_format in {"JPEG", "JPG"}:
        output = BytesIO(_encode_reference_jpeg(image, quality))
    elif normalized_format == "WEBP":
        converted = image
        if image.mode not in {"RGB", "RGBA"}:
            converted = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        try:
            converted.save(output, "WEBP", quality=quality, method=6)
        finally:
            if converted is not image:
                converted.close()
    else:
        raise ValueError(f"不支持保留的图片格式: {normalized_format or 'unknown'}")
    return output.getvalue()


def _compress_delivered_image_file(
    path,
    staging_dir,
    max_target_bytes,
    is_cancelled=lambda: False,
    allow_format_change=False,
):
    """Compress a fresh output, optionally converting it to a real JPEG."""
    source = os.path.abspath(str(path))
    source_size = os.path.getsize(source)
    with Image.open(source) as opened:
        source_format = str(opened.format or "").upper()
        if source_format not in {"PNG", "JPEG", "JPG", "WEBP"}:
            raise ValueError(f"不支持压缩的图片格式: {source_format or 'unknown'}")
        image = ImageOps.exif_transpose(opened)
        image.load()

    image_format = "JPEG" if allow_format_change else source_format

    width, height = image.size
    if width < 1 or height < 1:
        image.close()
        raise ValueError("生成图片尺寸无效")

    target_bytes = _reference_target_bytes_for_pixels(max_target_bytes, width * height)
    current_width, current_height = width, height
    # A fresh result has not been registered by the host yet, so returning a
    # new JPEG path is safe. Keep full dimensions unless JPEG quality alone
    # cannot satisfy the configured size limit.
    needs_resize = not allow_format_change and (
        max(width, height) > _REFERENCE_IMAGE_MAX_LONG_EDGE
        or width * height > _REFERENCE_IMAGE_SOFT_PIXELS
    )
    if needs_resize:
        scale = min(1.0, _REFERENCE_IMAGE_MAX_LONG_EDGE / max(width, height))
        if width * height > _REFERENCE_IMAGE_MAX_PIXELS:
            scale = min(scale, (_REFERENCE_IMAGE_MAX_PIXELS / (width * height)) ** 0.5)
        current_width = max(1, int(round(width * scale)))
        current_height = max(1, int(round(height * scale)))

    qualities = (92, 90, 88, 86, 84) if source_size / max(1, width * height) >= 2.5 else (90, 88, 86, 84, 82)
    best = None
    try:
        for _ in range(6):
            if is_cancelled():
                raise Exception("任务已被宿主取消；本地结果压缩已停止")
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            candidate_image = image.resize(
                (current_width, current_height), resampling
            ) if (current_width, current_height) != (width, height) else image.copy()
            try:
                candidates = []
                encode_qualities = (90,) if image_format == "PNG" else qualities
                for quality in encode_qualities:
                    candidate = _encode_delivered_image(candidate_image, image_format, quality)
                    candidates.append(candidate)
                    if best is None or len(candidate) < len(best):
                        best = candidate
                    if len(candidate) <= target_bytes:
                        break
            finally:
                candidate_image.close()
            if candidates and len(candidates[-1]) <= target_bytes:
                break
            if max(current_width, current_height) <= _REFERENCE_IMAGE_MIN_LONG_EDGE:
                break
            size_ratio = (
                (target_bytes / max(1, len(best))) ** 0.5 * 0.96
                if best
                else 0.72
            )
            size_ratio = max(0.55, min(0.90, size_ratio))
            current_width = max(1, int(current_width * size_ratio))
            current_height = max(1, int(current_height * size_ratio))
    finally:
        image.close()

    if not best or len(best) > target_bytes:
        raise ValueError(f"本地结果压缩失败: {os.path.basename(source)}")
    output_name = f"{Path(source).stem}.jpg" if allow_format_change else os.path.basename(source)
    output_path = os.path.join(staging_dir, output_name)
    with open(output_path, "wb") as output:
        output.write(best)
    return output_path


def _manual_replacement_format(suffix):
    normalized = str(suffix or "").lower()
    Image.init()
    common_formats = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
    }
    image_format = common_formats.get(normalized) or Image.registered_extensions().get(normalized)
    if not image_format or image_format not in Image.SAVE:
        raise Exception(f"不支持按原图后缀压缩: {suffix or '无后缀'}")
    return normalized, image_format


def _encode_manual_replacement_image(image, image_format):
    """Encode a static replacement without lying about its filename or MIME type."""
    buffer = BytesIO()
    if image_format == "PNG":
        image.save(buffer, "PNG", optimize=True, compress_level=9)
    elif image_format == "JPEG":
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, "JPEG", quality=90, optimize=True, progressive=True)
    elif image_format == "WEBP":
        image.save(buffer, "WEBP", quality=90, method=6)
    else:
        image.save(buffer, image_format)
    return buffer.getvalue()


def _compress_manual_replacement_output(path, original_suffix, params, is_cancelled=lambda: False):
    """Fit a manual replacement under its limit while retaining the host's original format."""
    source = Path(path)
    target_suffix, image_format = _manual_replacement_format(original_suffix)
    target_bytes = _local_result_target_bytes(params)
    needs_reencode = source.suffix.lower() != target_suffix
    if not source.is_file() or (source.stat().st_size <= target_bytes and not needs_reencode):
        return str(source)

    try:
        _local_reference_compression_gate.acquire(is_cancelled)
        try:
            with Image.open(source) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened)
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
                width, height = image.size
                current_width, current_height = width, height
                for _ in range(6):
                    if is_cancelled():
                        raise Exception("任务已被宿主取消；本地结果压缩已停止")
                    candidate = image.resize(
                        (current_width, current_height),
                        getattr(Image, "Resampling", Image).LANCZOS,
                    ) if (current_width, current_height) != (width, height) else image.copy()
                    encoded = _encode_manual_replacement_image(candidate, image_format)
                    candidate.close()
                    encoded_bytes = len(encoded)
                    temporary = source.with_name(
                        f".{source.stem}.compress-{uuid.uuid4().hex}{target_suffix}"
                    )
                    try:
                        temporary.write_bytes(encoded)
                        if encoded_bytes <= target_bytes:
                            print(
                                f"[Yali AI Image] 手动超分结果已压缩: {source.name} "
                                f"({encoded_bytes / 1024 / 1024:.2f} MB; {image_format})"
                            )
                            return str(temporary)
                    finally:
                        if temporary.exists() and encoded_bytes > target_bytes:
                            temporary.unlink(missing_ok=True)

                    if max(current_width, current_height) <= _REFERENCE_IMAGE_MIN_LONG_EDGE:
                        break
                    ratio = (target_bytes / max(1, encoded_bytes)) ** 0.5 * 0.98
                    ratio = max(0.72, min(0.94, ratio))
                    current_width = max(1, int(current_width * ratio))
                    current_height = max(1, int(current_height * ratio))
        finally:
            _local_reference_compression_gate.release()
    except Exception as exc:
        # A finished manual replacement must remain usable even when local
        # optimization cannot satisfy the configured size target.
        print(f"[Yali AI Image] 手动超分结果压缩跳过: {exc}")
    return str(source)


def _prepare_reference_images(reference_images, params, is_cancelled=lambda: False):
    """Prepare local references once and return (mapping, cleanup callback)."""
    normalized = _normalize_reference_images(reference_images)
    if not normalized:
        return {}, lambda: None

    target_bytes = _reference_target_bytes(params)
    local_paths = []
    for _, value in normalized.items():
        text = str(value or "").strip()
        if text and not text.lower().startswith(("http://", "https://")):
            if os.path.exists(text) and os.path.getsize(text) > 0:
                local_paths.append(os.path.abspath(text))

    if not local_paths:
        return normalized, lambda: None

    # Prepared references are reused by every concurrent storyboard item. Do
    # not create a staging directory when every source is within the limit.
    if all(
        os.path.getsize(path) <= _REFERENCE_IMAGE_THRESHOLD_BYTES
        for path in local_paths
    ):
        return normalized, lambda: None

    _local_reference_compression_gate.acquire(is_cancelled)
    try:
        staging_dir = tempfile.mkdtemp(prefix="yaliai-reference-")
        prepared = dict(normalized)
        try:
            for position, value in normalized.items():
                text = str(value or "").strip()
                if not text or text.lower().startswith(("http://", "https://")):
                    continue
                if os.path.exists(text) and os.path.getsize(text) > 0:
                    prepared[position] = _compress_reference_file(
                        text, staging_dir, target_bytes, is_cancelled=is_cancelled
                    )
        except Exception:
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    finally:
        _local_reference_compression_gate.release()

    def cleanup():
        import shutil
        shutil.rmtree(staging_dir, ignore_errors=True)

    return prepared, cleanup


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
        "name": "鸭梨AI图像生成插件(低价版)",
        "description": "通过鸭梨 AI 网关异步生成图片，支持 Gemini 原生接口、OpenAI Images 文件上传和任务日志。",
        "version": "3.3.0",
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


def handle_action(action, data=None, context=None):
    global _global_params

    if data is None:
        data = {}

    if action == "open_task_logs":
        return {"ok": True, "open_page": "task_log.html"}

    if action == "get_configured_references":
        return {"ok": True, "images": _configured_reference_info()}

    if action == "save_configured_references":
        return _save_configured_references(data.get("images", []))

    if action == "clear_configured_references":
        current = _configured_reference_paths()
        _remove_configured_reference_files(current)
        _save_config({"configured_reference_images": []})
        return {"ok": True, "images": []}

    if action == "save_param":
        key = data.get("key")
        value = data.get("value")

        if key is None:
            return {"ok": False, "error": "缺少 key"}

        # Normal iframe settings are persisted by PluginSDK.saveParam().
        # This bridge only keeps the currently loaded plugin module in sync;
        # writing a second config snapshot here can race with the host writer.
        if key == "model":
            value = _normalize_model(value)
        _global_params[key] = value
        return {"ok": True}

    elif action == "save_all_params":
        params = data.get("params", {})

        if not isinstance(params, dict):
            return {"ok": False, "error": "params 必须是 dict"}

        if "model" in params:
            params = dict(params)
            params["model"] = _normalize_model(params.get("model"))
        _global_params.update(params)
        return {"ok": True}

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

    elif action == "start_manual_upscale":
        return _start_manual_upscale(data.get("task_id"))

    elif action == "start_manual_upscale_batch":
        return _start_manual_upscale_batch(data.get("task_ids"))

    elif action == "open_local_task_image":
        return _open_local_task_image(data.get("task_id"))

    elif action == "set_task_frame":
        return _set_task_frame(data.get("task_id"), data.get("frame"))

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
            "generation_mode": "default",
            "pipeline_stage": "",
            "upscale_target_image_size": "",
            "upscale_applied": False,
            "source_model": "",
            "source_task_id": "",
            "viewer_index": 0,
            "unique_name": "",
            "generation_round": 0,
            "output_position": 0,
            "batch_index": 0,
            "batch_num": 1,
            "reference_image_count": 0,
            "shared_reference_prepare_ms": 0,
            "stage_reference_prepare_ms": 0,
            "submit_elapsed_ms": 0,
            "prompt_preview": "",
            "backup_path": "",
            "host_refresh_state": "",
            "host_refresh_reason": "",
            "host_refresh_image_index": None,
            "host_refresh_image_version": None,
            "host_frame": "",
            "host_frame_path": "",
            "local_image_format": "",
            "local_image_size_bytes": 0,
        })
        summary["created_at"] = min(summary["created_at"], timestamp) if timestamp else summary["created_at"]
        summary["updated_at"] = max(summary["updated_at"], timestamp)
        summary["event_count"] += 1
        summary["event"] = str(event.get("event", "") or summary["event"])
        for key in (
            "status", "error", "trace_id", "request_id", "query_path", "output_path", "output_type",
            "model", "quality", "image_size", "aspect_ratio", "request_size", "protocol", "generation_mode", "pipeline_stage",
            "upscale_target_image_size", "upscale_applied",
            "source_model", "source_task_id", "viewer_index", "unique_name",
            "generation_round", "output_position", "batch_index", "batch_num", "reference_image_count",
            "shared_reference_prepare_ms", "stage_reference_prepare_ms", "submit_elapsed_ms", "prompt_preview",
            "backup_path", "host_refresh_state", "host_refresh_reason", "host_refresh_image_index",
            "host_refresh_image_version", "host_frame", "host_frame_path",
            "local_image_format", "local_image_size_bytes",
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
    all_tasks = _summarize_async_task_events(_read_async_task_events())
    upscale_states = _build_manual_upscale_states(all_tasks)
    tasks = _group_upscale_workflows(all_tasks)
    if task_id:
        needle = task_id.lower()
        tasks = [
            item for item in tasks
            if needle in item["task_id"].lower()
            or needle in str((item.get("source_task") or {}).get("task_id", "")).lower()
        ]
    if status:
        tasks = [item for item in tasks if str(item.get("status", "")).lower() == status]
    total = len(tasks)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)
    start = (page - 1) * page_size
    page_tasks = tasks[start:start + page_size]
    for item in page_tasks:
        _attach_local_image_preview(item)
        source_task = item.get("source_task")
        if isinstance(source_task, dict):
            _attach_local_image_preview(source_task)
        state = upscale_states.get(str(item.get("task_id", "")), "unavailable")
        item["upscale_state"] = state
        item["already_upscaled"] = state == "already_upscaled"
        item["can_manual_upscale"] = state == "eligible" and bool(item.get("local_image_exists"))
    return {
        "tasks": page_tasks,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
    }


def _group_upscale_workflows(tasks):
    """Collapse automatic and manual A/B upscale stages into one workflow row."""
    by_task_id = {str(item.get("task_id", "")): item for item in tasks}
    hidden_source_ids = set()
    manual_by_source = {}
    manual_child_ids = set()
    for item in tasks:
        source_id = str(item.get("source_task_id", "") or "")
        if (
            str(item.get("generation_mode", "")).lower() == "manual_upscale"
            and source_id in by_task_id
        ):
            task_id = str(item.get("task_id", ""))
            manual_child_ids.add(task_id)
            rank = (
                _safe_int(item.get("updated_at", 0), 0),
                _safe_int(item.get("event_count", 0), 0),
            )
            current = manual_by_source.get(source_id)
            if current is None or rank >= current[0]:
                manual_by_source[source_id] = (rank, item)

    grouped = []
    for item in tasks:
        task_id = str(item.get("task_id", ""))
        source_id = str(item.get("source_task_id", "") or "")
        is_auto_upscale = (
            str(item.get("generation_mode", "")).lower() == "upscale"
            and str(item.get("pipeline_stage", "")).lower() == "upscale"
            and source_id in by_task_id
        )
        is_current_manual = (
            source_id in manual_by_source
            and manual_by_source[source_id][1] is item
        )
        if is_auto_upscale:
            combined = dict(item)
            combined["workflow_type"] = "upscale"
            combined["source_task"] = dict(by_task_id[source_id])
            grouped.append(combined)
            hidden_source_ids.add(source_id)
        elif is_current_manual:
            combined = dict(item)
            combined["workflow_type"] = "manual_upscale"
            combined["source_task"] = dict(by_task_id[source_id])
            grouped.append(combined)
            hidden_source_ids.add(source_id)
        elif task_id in manual_child_ids:
            # A manual run first has a local queue receipt, then the gateway's
            # task ID. Present only the newest stage as one workflow row.
            continue
        else:
            grouped.append(item)
    visible = [item for item in grouped if str(item.get("task_id", "")) not in hidden_source_ids]
    return sorted(visible, key=lambda item: _safe_int(item.get("updated_at", 0), 0), reverse=True)


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
            if before_timestamp is None:
                shutil.rmtree(_TASK_THUMBNAIL_DIR, ignore_errors=True)
            else:
                _trim_task_thumbnails()
            return removed
        except OSError:
            return 0


def _trim_task_thumbnails(max_files=500):
    try:
        if not _TASK_THUMBNAIL_DIR.exists():
            return
        files = [item for item in _TASK_THUMBNAIL_DIR.glob("*.jpg") if item.is_file()]
        files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        cutoff = time.time() - 30 * 86400
        for index, item in enumerate(files):
            if index >= max_files or item.stat().st_mtime < cutoff:
                item.unlink(missing_ok=True)
    except OSError:
        pass


def _task_thumbnail_path(task_id):
    key = hashlib.sha256(f"task:{task_id}".encode("utf-8")).hexdigest()
    return _TASK_THUMBNAIL_DIR / f"task_{key}.jpg"


def _thumbnail_data_url(thumbnail):
    return "data:image/jpeg;base64," + base64.b64encode(thumbnail).decode("ascii")


def _persist_task_thumbnail(task_id, path):
    """Store a small local receipt before an A-stage temporary image is removed."""
    source = Path(path)
    if not task_id or not source.is_file():
        return ""
    try:
        _TASK_THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _task_thumbnail_path(task_id)
        if cache_path.exists():
            return _thumbnail_data_url(cache_path.read_bytes())
        with Image.open(source) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((192, 144), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, "JPEG", quality=80, optimize=True)
            thumbnail = buffer.getvalue()
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_bytes(thumbnail)
        os.replace(temp_path, cache_path)
        _trim_task_thumbnails()
        return _thumbnail_data_url(thumbnail)
    except Exception:
        return ""


def _attach_local_image_preview(summary):
    path = Path(str(summary.get("output_path", "") or ""))
    try:
        summary["local_image_exists"] = path.is_file() and path.stat().st_size > 0
    except OSError:
        summary["local_image_exists"] = False
    summary["local_preview_data_url"] = ""
    task_id = str(summary.get("task_id", "") or "")
    try:
        cache_path = _task_thumbnail_path(task_id) if task_id else None
        if cache_path and cache_path.exists():
            summary["local_preview_data_url"] = _thumbnail_data_url(cache_path.read_bytes())
        elif summary["local_image_exists"]:
            summary["local_preview_data_url"] = _persist_task_thumbnail(task_id, path)
    except Exception as exc:
        summary["local_preview_error"] = str(exc)


def _canonical_output_path(value):
    try:
        return str(Path(str(value or "")).resolve()).lower()
    except OSError:
        return ""


def _is_upscale_result(summary):
    return (
        str(summary.get("generation_mode", "")).lower() in {"upscale", "manual_upscale"}
        or str(summary.get("pipeline_stage", "")).lower() in {"upscale", "manual_upscale"}
    )


def _build_manual_upscale_states(tasks):
    """Only the newest successful result for a local asset may be manually upscaled once."""
    newest_by_path = {}
    for summary in tasks:
        if str(summary.get("status", "")).lower() != "success":
            continue
        path_key = _canonical_output_path(summary.get("output_path"))
        if not path_key:
            continue
        current = newest_by_path.get(path_key)
        rank = (_safe_int(summary.get("updated_at", 0), 0), _safe_int(summary.get("event_count", 0), 0))
        if current is None or rank >= current[0]:
            newest_by_path[path_key] = (rank, str(summary.get("task_id", "")), summary)

    states = {}
    for summary in tasks:
        task_id = str(summary.get("task_id", ""))
        if str(summary.get("status", "")).lower() != "success":
            states[task_id] = "unavailable"
            continue
        path_key = _canonical_output_path(summary.get("output_path"))
        latest = newest_by_path.get(path_key)
        if not path_key or latest is None:
            states[task_id] = "unavailable"
        elif latest[1] != task_id:
            states[task_id] = "superseded"
        elif _is_upscale_result(summary):
            states[task_id] = "already_upscaled"
        else:
            states[task_id] = "eligible"
    return states


def _open_local_task_image(task_id):
    summary = _find_task_log_summary(task_id)
    if not summary:
        return {"ok": False, "error": "未找到任务记录"}
    path = Path(str(summary.get("output_path", "") or ""))
    if not path.is_file() or path.stat().st_size <= 0:
        return {"ok": False, "error": "本地图片文件不存在"}
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        return {"ok": False, "error": f"本地图片无法读取: {exc}"}
    try:
        if os.name == "nt":
            os.startfile(str(path))
        else:
            return {"ok": False, "error": "当前系统不支持从插件打开本地图片"}
    except OSError as exc:
        return {"ok": False, "error": f"无法打开本地图片: {exc}"}
    return {"ok": True, "message": "已使用系统默认图片查看器打开本地图片"}


def _set_task_frame(task_id, frame):
    """Use a delivered task image as the explicitly requested storyboard keyframe."""
    summary = _find_task_log_summary(task_id)
    if not summary:
        return {"ok": False, "error": "未找到任务记录"}
    if str(summary.get("status", "")).lower() != "success":
        return {"ok": False, "error": "仅已交付的图片可以设为首尾帧"}

    frame = str(frame or "").strip().lower()
    tool_name = {
        "first": "zzdh_set_first_frame",
        "end": "zzdh_set_end_frame",
    }.get(frame)
    if not tool_name:
        return {"ok": False, "error": "frame 只能是 first 或 end"}

    unique_name = str(summary.get("unique_name", "") or "").strip()
    if not unique_name:
        return {"ok": False, "error": "该任务没有可关联的分镜"}
    path = Path(str(summary.get("output_path", "") or ""))
    if not path.is_file() or path.stat().st_size <= 0:
        return {"ok": False, "error": "本地图片文件不存在"}
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        return {"ok": False, "error": f"本地图片无法读取: {exc}"}

    try:
        _call_host_tool(tool_name, {
            "unique_name": unique_name,
            "image_path": str(path.resolve()),
        })
    except Exception as exc:
        return {"ok": False, "error": f"写入宿主{('首帧' if frame == 'first' else '尾帧')}失败: {exc}"}

    label = "首帧" if frame == "first" else "尾帧"
    _record_async_task(
        "host_frame_linked",
        task_id=str(summary["task_id"]),
        status="success",
        host_frame=frame,
        host_frame_path=str(path.resolve()),
    )
    return {"ok": True, "message": f"已设为分镜{label}"}


def _file_fingerprint(path):
    """Return a content fingerprint so an old log row cannot overwrite a changed image."""
    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256()
    with open(source, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{stat.st_size}:{stat.st_mtime_ns}:{digest.hexdigest()}"


def _same_local_path(left, right):
    """Compare host file paths without depending on their spelling or case."""
    try:
        return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(os.path.abspath(os.fspath(right)))
    except (TypeError, ValueError, OSError):
        return False


def _call_host_tool(name, arguments):
    """Call the host's documented local tools endpoint, never a private host API."""
    endpoint = str(os.environ.get("YALIAI_HOST_TOOL_CALL_ENDPOINT", _HOST_TOOL_CALL_ENDPOINT) or "").strip()
    if not endpoint:
        raise RuntimeError("宿主工具端点未配置")
    response = requests.post(
        endpoint,
        json={"name": str(name), "arguments": dict(arguments or {})},
        timeout=_HOST_TOOL_CALL_TIMEOUT_SECONDS,
    )
    status_code = _safe_int(getattr(response, "status_code", 0), 0)
    if status_code < 200 or status_code >= 300:
        raise RuntimeError(f"宿主工具 HTTP {status_code}: {str(getattr(response, 'text', ''))[:240]}")
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"宿主工具返回非 JSON 数据: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("宿主工具返回格式无效")
    if payload.get("success") is False or payload.get("ok") is False:
        raise RuntimeError(str(payload.get("error") or "宿主工具执行失败"))
    return payload


def _refresh_host_image_after_manual_upscale(source, unique_name):
    """Refresh only the host's still-selected source asset after an in-place replacement.

    ``zzdh_import_image_from_path`` is the documented host operation that updates
    the media version and broadcasts a refresh to the canvas. It has no image
    index argument, so importing an arbitrary older task could replace the wrong
    slot. Requiring an exact match with the selected asset avoids that race.
    """
    source = Path(source)
    unique_name = str(unique_name or "").strip()
    if not unique_name:
        return {"state": "skipped", "reason": "任务缺少分镜标识"}
    if not source.is_file():
        return {"state": "skipped", "reason": "超分结果文件不存在"}

    try:
        before = _call_host_tool("zzdh_get_edit_view_data", {"unique_name": unique_name})
        images = before.get("images")
        selected_index = _safe_int(before.get("selected_index", -1), -1)
        if not isinstance(images, list) or selected_index < 0 or selected_index >= len(images):
            return {"state": "skipped", "reason": "宿主未找到当前选中图片"}
        selected = images[selected_index] if isinstance(images[selected_index], dict) else {}
        selected_path = selected.get("path", "")
        if not _same_local_path(selected_path, source):
            return {
                "state": "skipped",
                "reason": "分镜当前图片已变化，未覆盖后续结果",
                "selected_path": str(selected_path or ""),
            }

        refreshed = _call_host_tool(
            "zzdh_import_image_from_path",
            {"unique_name": unique_name, "image_path": str(source.resolve())},
        )
        refreshed_index = _safe_int(refreshed.get("image_index", -1), -1)
        if refreshed_index != selected_index:
            raise RuntimeError("宿主刷新返回了非当前图片槽位")
        after = _call_host_tool("zzdh_get_edit_view_data", {"unique_name": unique_name})
        after_images = after.get("images")
        if not isinstance(after_images, list) or refreshed_index >= len(after_images):
            raise RuntimeError("宿主刷新后未返回目标图片")
        updated = after_images[refreshed_index] if isinstance(after_images[refreshed_index], dict) else {}
        return {
            "state": "refreshed",
            "image_index": refreshed_index,
            "image_version": updated.get("version"),
            "path": str(updated.get("path") or ""),
        }
    except Exception as exc:
        # The image replacement has already succeeded. A disconnected host must
        # not turn that completed, billable upstream job into a failed task.
        return {"state": "unavailable", "reason": str(exc)}


def _find_task_log_summary(task_id):
    wanted = str(task_id or "").strip()
    if not wanted:
        return None
    for item in _summarize_async_task_events(_read_async_task_events()):
        if str(item.get("task_id", "")) == wanted:
            return item
    return None


def _infer_aspect_ratio_from_image(path, fallback):
    """Use the actual local source dimensions, not a possibly stale log value."""
    try:
        with Image.open(path) as image:
            width, height = image.size
        if width > 0 and height > 0:
            ratio = width / height
            return min(_SUPPORTED_ASPECT_RATIOS, key=lambda key: abs(_SUPPORTED_ASPECT_RATIOS[key] - ratio))
    except Exception:
        pass
    fallback = str(fallback or "16:9").strip()
    return fallback if fallback in _SUPPORTED_ASPECT_RATIOS else "16:9"


def _manual_upscale_metadata(summary, params, source_task_id, source_path):
    upscale_model = _normalize_model(params.get("upscale_model", "gemini-3-pro-image-preview"))
    image_size = _normalize_image_size(params.get("upscale_image_size", "4K"), "4K")
    aspect_ratio = _infer_aspect_ratio_from_image(
        source_path, summary.get("aspect_ratio") or params.get("aspect_ratio", "16:9")
    )
    return {
        "model": upscale_model,
        "quality": str(params.get("quality", "medium") or "medium").strip().lower(),
        "image_size": image_size,
        "aspect_ratio": aspect_ratio,
        "protocol": "openai_image" if upscale_model in GPT_IMAGE_MODELS else "gemini",
        "generation_mode": "manual_upscale",
        "pipeline_stage": "manual_upscale",
        "upscale_target_image_size": image_size,
        "upscale_applied": True,
        "source_model": str(summary.get("model", "") or ""),
        "source_task_id": source_task_id,
        "viewer_index": _safe_int(summary.get("viewer_index", 0), 0),
        "unique_name": str(summary.get("unique_name", "") or ""),
        "generation_round": _safe_int(summary.get("generation_round", 0), 0),
        "output_position": _safe_int(summary.get("output_position", 0), 0),
        "batch_index": 0,
        "batch_num": 1,
        "reference_image_count": 1,
        "stage_max_wait": _ASYNC_MAX_WAIT_SECONDS,
        "workflow_max_wait": _ASYNC_MAX_WAIT_SECONDS,
        "prompt_preview": _prompt_preview(_render_upscale_prompt(
            params.get("upscale_prompt", _DEFAULT_UPSCALE_PROMPT), image_size, aspect_ratio
        )),
    }


def _run_manual_upscale(summary, source_path, source_key, local_job_id):
    """Run B-stage-only upscaling and replace the host asset only after a full download succeeds."""
    source_task_id = str(summary["task_id"])
    gateway_task_id = local_job_id
    prepared_cleanup = lambda: None
    reference_slot = False
    task_slot = False
    delivery_slot = False
    try:
        source = Path(source_path)
        original_fingerprint = _file_fingerprint(source)
        params = _merge_runtime_params({})
        endpoint = _normalize_endpoint(params.get("endpoint", _GATEWAY_ENDPOINT))
        metadata = _manual_upscale_metadata(summary, params, source_task_id, source)
        _record_async_task(
            "manual_upscale_processing",
            task_id=local_job_id,
            status="processing",
            **metadata,
        )
        model = metadata["model"]
        api_key = _api_key_for_model(params, model)
        if not api_key:
            label = "GPT KEY" if model in GPT_IMAGE_MODELS else "GEMINI KEY"
            raise Exception(f"未设置超分模型所需的 {label}")

        _local_reference_gate.acquire(lambda: False)
        reference_slot = True
        _local_task_gate.acquire(lambda: False)
        task_slot = True
        prepared_references, prepared_cleanup = _prepare_reference_images({0: str(source)}, params)
        prompt = _render_upscale_prompt(
            params.get("upscale_prompt", _DEFAULT_UPSCALE_PROMPT),
            metadata["image_size"], metadata["aspect_ratio"],
        )
        request_id = f"yaliai_plugin_manual_upscale_{uuid.uuid4().hex}"
        if model in GPT_IMAGE_MODELS:
            outputs = send_gpt_image_request(
                api_key=api_key,
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                reference_images=prepared_references,
                aspect_ratio=metadata["aspect_ratio"],
                image_size=metadata["image_size"],
                quality=metadata["quality"],
                request_timeout=_GATEWAY_HTTP_TIMEOUT_SECONDS,
                download_timeout=_IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
                async_initial_delay=_ASYNC_INITIAL_DELAY_SECONDS,
                async_poll_interval=_ASYNC_POLL_INTERVAL_SECONDS,
                async_max_wait=_ASYNC_MAX_WAIT_SECONDS,
                request_id=request_id,
                task_metadata=metadata,
            )
        else:
            outputs = send_gemini_request(
                api_key=api_key,
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                reference_images=prepared_references,
                aspect_ratio=metadata["aspect_ratio"],
                image_size=metadata["image_size"],
                request_timeout=_GATEWAY_HTTP_TIMEOUT_SECONDS,
                download_timeout=_IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
                async_initial_delay=_ASYNC_INITIAL_DELAY_SECONDS,
                async_poll_interval=_ASYNC_POLL_INTERVAL_SECONDS,
                async_max_wait=_ASYNC_MAX_WAIT_SECONDS,
                request_id=request_id,
                task_metadata=metadata,
            )
        if not outputs:
            raise Exception("超分任务完成但没有返回图片")

        output = outputs[0]
        gateway_task_id = str(output.get("task_id") or gateway_task_id)
        with tempfile.TemporaryDirectory(prefix="yaliai-manual-upscale-", dir=str(source.parent)) as staging_dir:
            temp_context = {
                "viewer_index": 0,
                "unique_name": f"manual_upscale_{uuid.uuid4().hex[:12]}",
                "generation_round": int(time.time()),
                "output_position": [0],
            }
            _local_delivery_gate.acquire(lambda: False)
            delivery_slot = True
            try:
                if output.get("type") == "url":
                    output_url = _absolute_gateway_url(endpoint, output.get("value"))
                    staged_path = download_url_to_output(
                        output_url, temp_context, staging_dir,
                        download_timeout=_IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
                    )
                elif output.get("type") == "b64":
                    output_url = ""
                    staged_path = save_image_base64_to_output(output.get("value"), temp_context, staging_dir)
                else:
                    raise Exception("超分结果包含未知图片格式")
            finally:
                _local_delivery_gate.release()
                delivery_slot = False

            staged_path = _compress_manual_replacement_output(
                staged_path, source.suffix, params
            )
            if _file_fingerprint(source) != original_fingerprint:
                raise Exception("原图在超分期间已变化，已取消替换以避免覆盖新图片")
            backup_dir = source.parent / ".yaliai-backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{source.stem}_{int(time.time())}_{uuid.uuid4().hex[:8]}{source.suffix}"
            shutil.copy2(source, backup_path)
            os.replace(staged_path, source)

        host_refresh = _refresh_host_image_after_manual_upscale(
            source, metadata.get("unique_name")
        )
        _record_async_task(
            "manual_upscale_replaced",
            task_id=gateway_task_id,
            status="success",
            output_path=str(source.resolve()),
            output_url=output_url,
            output_type=output.get("type", ""),
            backup_path=str(backup_path.resolve()),
            host_refresh_state=host_refresh.get("state", ""),
            host_refresh_reason=host_refresh.get("reason", ""),
            host_refresh_image_index=host_refresh.get("image_index"),
            host_refresh_image_version=host_refresh.get("image_version"),
            **metadata,
        )
        _persist_task_thumbnail(gateway_task_id, source)
        if host_refresh.get("state") == "refreshed":
            print(f"[Yali AI Image] 超分已替换并刷新分镜: {source}")
        else:
            print(f"[Yali AI Image] 超分已替换，分镜未刷新: {source} ({host_refresh.get('reason', '')})")
    except Exception as exc:
        _record_async_task(
            "manual_upscale_failed",
            task_id=gateway_task_id,
            status="failed",
            error=str(exc),
            source_task_id=source_task_id,
            pipeline_stage="manual_upscale",
            generation_mode="manual_upscale",
        )
        print(f"[Yali AI Image] 手动超分失败: {exc}")
    finally:
        try:
            prepared_cleanup()
        except Exception:
            pass
        if delivery_slot:
            _local_delivery_gate.release()
        if task_slot:
            _local_task_gate.release()
        if reference_slot:
            _local_reference_gate.release()
        with _manual_upscale_jobs_lock:
            _manual_upscale_jobs.discard(source_key)


def _start_manual_upscale(task_id):
    summary = _find_task_log_summary(task_id)
    if not summary:
        return {"ok": False, "error": "未找到任务记录"}
    upscale_state = _build_manual_upscale_states(_summarize_async_task_events(_read_async_task_events())).get(
        str(summary.get("task_id", "")), "unavailable"
    )
    if upscale_state == "already_upscaled":
        return {"ok": False, "error": "该图片已超分，不允许重复超分"}
    if upscale_state == "superseded":
        return {"ok": False, "error": "该任务图片已被后续结果替换，请选择最新任务"}
    if upscale_state != "eligible":
        return {"ok": False, "error": "仅当前未超分的成功图片可以超分"}
    source = Path(str(summary.get("output_path", "") or ""))
    if not source.is_file() or source.stat().st_size <= 0:
        return {"ok": False, "error": "原图文件不存在，无法替换对应分镜"}
    try:
        with Image.open(source) as image:
            image.verify()
    except Exception as exc:
        return {"ok": False, "error": f"原图无法读取: {exc}"}
    source_key = str(source.resolve()).lower()
    with _manual_upscale_jobs_lock:
        if source_key in _manual_upscale_jobs:
            return {"ok": False, "error": "该图片已有超分任务正在执行"}
        _manual_upscale_jobs.add(source_key)
    local_job_id = f"local_upscale_{uuid.uuid4().hex}"
    try:
        pending_params = _merge_runtime_params({})
        pending_metadata = _manual_upscale_metadata(
            summary, pending_params, str(summary["task_id"]), source
        )
        _record_async_task(
            "manual_upscale_queued",
            task_id=local_job_id,
            status="queued",
            **pending_metadata,
        )
    except Exception:
        with _manual_upscale_jobs_lock:
            _manual_upscale_jobs.discard(source_key)
        raise
    worker = threading.Thread(
        target=_run_manual_upscale,
        args=(dict(summary), str(source.resolve()), source_key, local_job_id),
        name=f"yaliai-manual-upscale-{uuid.uuid4().hex[:8]}",
        daemon=True,
    )
    worker.start()
    return {
        "ok": True,
        "task_id": local_job_id,
        "source_task_id": str(summary["task_id"]),
        "message": "超分任务已进入队列；完成后会备份并替换当前分镜图片",
    }


def _start_manual_upscale_batch(task_ids):
    if not isinstance(task_ids, (list, tuple)):
        return {"ok": False, "error": "task_ids 必须是任务 ID 列表"}
    unique_ids = []
    seen = set()
    for value in task_ids:
        task_id = str(value or "").strip()
        if task_id and task_id not in seen:
            seen.add(task_id)
            unique_ids.append(task_id)
    if not unique_ids:
        return {"ok": False, "error": "请先选择已交付的任务"}
    if len(unique_ids) > _MANUAL_UPSCALE_BATCH_LIMIT:
        return {"ok": False, "error": f"一次最多批量超分 {_MANUAL_UPSCALE_BATCH_LIMIT} 张图片"}

    started = []
    rejected = []
    for task_id in unique_ids:
        result = _start_manual_upscale(task_id)
        if result.get("ok"):
            started.append(task_id)
        else:
            rejected.append({"task_id": task_id, "error": result.get("error", "无法提交")})
    if not started:
        message = rejected[0]["error"] if rejected else "没有可提交的超分任务"
        return {"ok": False, "error": message, "rejected": rejected}
    return {
        "ok": True,
        "started": started,
        "rejected": rejected,
        "message": f"已提交 {len(started)} 个超分任务；将按本地并发队列执行并替换原图",
    }


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
    submitted_at = time.monotonic()
    if json_payload is not None:
        headers["Content-Type"] = "application/json"
        response = session.post(url, headers=headers, json=json_payload, timeout=request_timeout)
    else:
        response = session.post(url, headers=headers, data=form_data, files=files, timeout=request_timeout)
    submit_elapsed_ms = int((time.monotonic() - submitted_at) * 1000)

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
        submit_elapsed_ms=submit_elapsed_ms,
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

    initial_delay = max(1, int(initial_delay))
    poll_interval = max(1, int(poll_interval))
    max_wait = max(60, int(max_wait))
    progress("已提交", 15)
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
            progress("下载中", 85)
            return outputs
        if status in {"failed", "cancelled", "canceled", "expired"}:
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            message = error.get("message") or error.get("code") or payload.get("message") or status
            _record_async_task("failed", task_id=task_id, status=status, error=str(message))
            raise Exception(f"鸭梨 AI 任务{status}: {message}")

        elapsed = max_wait - max(0, int(deadline - time.monotonic()))
        progress("排队中" if status in {"queued", "pending"} else "生成中", min(84, 20 + int(elapsed * 60 / max(1, max_wait))))
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
    output_format=_OPENAI_IMAGE_OUTPUT_FORMAT,
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
    output_format = str(output_format or _OPENAI_IMAGE_OUTPUT_FORMAT).strip().lower()
    if output_format == "jpg":
        output_format = "jpeg"
    if output_format not in {"jpeg", "png", "webp"}:
        raise Exception("OpenAI Images output_format 必须是 jpeg、png 或 webp")
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
        "output_format": output_format,
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
    """Generate one host task, or a host-provided batch, in deterministic order.

    Batch items are submitted and polled concurrently. Each worker owns its
    HTTP session; results are sorted by output position before returning.
    """
    context = context or {}
    params = _merge_runtime_params(context)
    prompt = str(context.get("prompt", "") or "").strip()
    reference_images = context.get("reference_images", {}) or {}
    output_dir = _get_output_dir(context)
    endpoint = _normalize_endpoint(params.get("endpoint", _GATEWAY_ENDPOINT))
    model = _normalize_model(params.get("model", "gemini-3.1-flash-image-preview"))
    api_key = _api_key_for_model(params, model)
    aspect_ratio = str(params.get("aspect_ratio", "16:9") or "16:9").strip()
    image_size = _normalize_image_size(params.get("image_size", "4K"), "4K")
    quality = str(params.get("quality", "medium") or "medium").strip().lower()
    generation_mode = str(params.get("generation_mode", "default") or "default").strip().lower()
    upscale_model = _normalize_model(params.get("upscale_model", "gemini-3-pro-image-preview"))
    upscale_image_size = _normalize_image_size(params.get("upscale_image_size", "4K"), "4K")
    upscale_prompt_template = str(
        params.get("upscale_prompt", _DEFAULT_UPSCALE_PROMPT) or _DEFAULT_UPSCALE_PROMPT
    ).strip()
    request_timeout = _GATEWAY_HTTP_TIMEOUT_SECONDS
    download_timeout = _IMAGE_DOWNLOAD_TIMEOUT_SECONDS
    initial_delay = _ASYNC_INITIAL_DELAY_SECONDS
    poll_interval = _ASYNC_POLL_INTERVAL_SECONDS
    max_wait = _ASYNC_MAX_WAIT_SECONDS
    should_upscale = generation_mode == "upscale"
    workflow_max_wait = max_wait * (2 if should_upscale else 1)
    batch_num = _safe_int(context.get("batch_num", 1), 1)
    if batch_num < 1 or batch_num > _MAX_BATCH_NUM:
        raise Exception(f"PLUGIN_ERROR:::batch_num 必须在 1 到 {_MAX_BATCH_NUM} 之间")
    output_positions = context.get("output_position")
    if not isinstance(output_positions, (list, tuple)):
        output_positions = []

    configured_references = _configured_reference_paths()
    reference_images = _merge_reference_images(reference_images, configured_references)

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
    if generation_mode not in {"default", "upscale"}:
        raise Exception("PLUGIN_ERROR:::生成模式必须是 default 或 upscale")
    if generation_mode == "upscale":
        if should_upscale and not _api_key_for_model(params, upscale_model):
            credential_label = "GPT-image-2 API Key" if upscale_model in GPT_IMAGE_MODELS else "Gemini API Key"
            raise Exception(f"PLUGIN_ERROR:::未设置超分模型所需的 {credential_label}")
        _render_upscale_prompt(upscale_prompt_template, upscale_image_size, aspect_ratio)

    os.makedirs(output_dir, exist_ok=True)
    progress_callback = context.get("progress_callback")
    progress_lock = threading.Lock()
    progress_percent = 0

    def progress(text, percent=None):
        nonlocal progress_percent
        if not progress_callback:
            return
        with progress_lock:
            if percent is not None:
                percent = max(progress_percent, min(100, int(percent)))
                progress_percent = percent
            if percent is None:
                callback_args = (text,)
            else:
                callback_args = (text, percent)
            try:
                progress_callback(*callback_args)
                return
            except TypeError:
                pass
            except Exception:
                return
            try:
                progress_callback(text)
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("鸭梨 AI 图像生成插件开始异步任务")
    print("=" * 60)
    print(f"模型: {model}; 比例: {aspect_ratio}; 档位: {image_size}; 画质: {quality}; 批次: {batch_num}")
    if generation_mode == "upscale":
        print(f"超分规则: 基础图={image_size}; 目标={upscale_image_size}; 本次执行超分")
    print(f"参考图数量: {len(_normalize_reference_images(reference_images))}")

    def is_cancelled():
        return _is_cancelled(context)

    generated_files = [None] * batch_num
    progress("准备参考图", 6)
    shared_reference_prepare_started = time.monotonic()
    reference_images, cleanup_reference_images = _prepare_reference_images(reference_images, params)
    shared_reference_prepare_ms = int((time.monotonic() - shared_reference_prepare_started) * 1000)
    reference_image_count = len(_normalize_reference_images(reference_images))

    def stage_progress(start_percent, end_percent):
        def callback(text, percent=None):
            if percent is None:
                progress(text)
                return
            ratio = max(0, min(100, int(percent))) / 100
            progress(text, int(start_percent + (end_percent - start_percent) * ratio))

        return callback

    def execute_one(index):
        position = output_positions[index] if index < len(output_positions) else index
        source_task_metadata = {
            "model": model,
            "quality": quality,
            "image_size": image_size,
            "aspect_ratio": aspect_ratio,
            "protocol": "openai_image" if model in GPT_IMAGE_MODELS else "gemini",
            "generation_mode": generation_mode,
            "pipeline_stage": "source" if should_upscale else "single",
            "upscale_target_image_size": upscale_image_size,
            "upscale_applied": should_upscale,
            "source_model": model if should_upscale else "",
            "source_task_id": "",
            "stage_max_wait": max_wait,
            "workflow_max_wait": workflow_max_wait,
            "viewer_index": _safe_int(context.get("viewer_index", 0), 0),
            "unique_name": str(context.get("unique_name", "") or ""),
            "generation_round": _safe_int(context.get("generation_round", 0), 0),
            "output_position": position,
            "batch_index": index,
            "batch_num": batch_num,
            "reference_image_count": reference_image_count,
            "shared_reference_prepare_ms": shared_reference_prepare_ms,
            "prompt_preview": _prompt_preview(prompt),
        }
        if is_cancelled():
            raise Exception("任务已被宿主取消")

        def request_stage(stage_model, stage_prompt, stage_references, stage_request_id, stage_metadata, start_percent, end_percent, stage_image_size, references_prepared=False):
            if is_cancelled():
                raise Exception("任务已被宿主取消")
            preparation_started = time.monotonic()
            progress("准备参考图", max(6, start_percent - 2))
            if references_prepared:
                prepared_references, cleanup_stage_references = stage_references, (lambda: None)
            else:
                prepared_references, cleanup_stage_references = _prepare_reference_images(
                    stage_references, params, is_cancelled=is_cancelled
                )
            stage_metadata = dict(stage_metadata)
            stage_metadata["stage_reference_prepare_ms"] = int((time.monotonic() - preparation_started) * 1000)
            progress("提交任务", start_percent)
            try:
                if stage_model in GPT_IMAGE_MODELS:
                    outputs = send_gpt_image_request(
                        api_key=_api_key_for_model(params, stage_model),
                        endpoint=endpoint,
                        model=stage_model,
                        prompt=stage_prompt,
                        reference_images=prepared_references,
                        aspect_ratio=aspect_ratio,
                        image_size=stage_image_size,
                        quality=quality,
                        request_timeout=request_timeout,
                        download_timeout=download_timeout,
                        async_initial_delay=initial_delay,
                        async_poll_interval=poll_interval,
                        async_max_wait=max_wait,
                        progress=stage_progress(start_percent, end_percent),
                        request_id=stage_request_id,
                        is_cancelled=is_cancelled,
                        task_metadata=stage_metadata,
                    )
                else:
                    outputs = send_gemini_request(
                        api_key=_api_key_for_model(params, stage_model),
                        endpoint=endpoint,
                        model=stage_model,
                        prompt=stage_prompt,
                        reference_images=prepared_references,
                        aspect_ratio=aspect_ratio,
                        image_size=stage_image_size,
                        request_timeout=request_timeout,
                        download_timeout=download_timeout,
                        async_initial_delay=initial_delay,
                        async_poll_interval=poll_interval,
                        async_max_wait=max_wait,
                        progress=stage_progress(start_percent, end_percent),
                        request_id=stage_request_id,
                        is_cancelled=is_cancelled,
                        task_metadata=stage_metadata,
                    )
            finally:
                cleanup_stage_references()
            if not outputs:
                raise Exception(f"第 {index + 1} 个鸭梨 AI 任务完成但没有图片输出")
            if len(outputs) > 1:
                print(f"警告：任务返回 {len(outputs)} 张图片；当前宿主槽位只接收第一张")
            return outputs[0]

        def deliver_stage(output, target_dir, event_name, stage_metadata, target_position, compress_result=False):
            task_id = str(output.get("task_id", "") or "")
            image_url = ""
            delivery_slot = False
            progress("下载中")
            try:
                _local_delivery_gate.acquire(is_cancelled)
                delivery_slot = True
                if output.get("type") == "url":
                    image_url = _absolute_gateway_url(endpoint, output.get("value"))
                    if not image_url:
                        raise Exception("任务结果缺少图片 URL")
                    path = download_url_to_output(
                        image_url,
                        context,
                        target_dir,
                        download_timeout=download_timeout,
                        position_override=target_position,
                        is_cancelled=is_cancelled,
                    )
                elif output.get("type") == "b64":
                    path = save_image_base64_to_output(output.get("value"), context, target_dir, target_position)
                else:
                    raise Exception("任务结果包含未知图片格式")
            except Exception as delivery_error:
                _record_async_task(
                    "delivery_failed",
                    task_id=task_id,
                    status="download_failed",
                    error=str(delivery_error),
                    **stage_metadata,
                )
                raise
            finally:
                if delivery_slot:
                    _local_delivery_gate.release()
            if compress_result:
                path = _compress_delivered_output(
                    path,
                    params,
                    is_cancelled=is_cancelled,
                    allow_format_change=True,
                )
            _persist_task_thumbnail(task_id, path)
            try:
                with Image.open(path) as delivered_image:
                    local_image_format = str(delivered_image.format or "").lower()
            except Exception:
                local_image_format = ""
            _record_async_task(
                event_name,
                task_id=task_id,
                status="success" if event_name == "delivered" else "intermediate",
                output_path=os.path.abspath(path),
                output_url=image_url,
                output_type=output.get("type", ""),
                local_image_format=local_image_format,
                local_image_size_bytes=os.path.getsize(path),
                **stage_metadata,
            )
            return path, task_id

        if should_upscale:
            with tempfile.TemporaryDirectory(prefix="yaliai-upscale-") as staging_dir:
                source_output = request_stage(
                    model,
                    prompt,
                    reference_images,
                    _new_request_id(context, position),
                    source_task_metadata,
                    8 + int(index * 35 / batch_num),
                    42,
                    image_size,
                    references_prepared=True,
                )
                source_path, source_task_id = deliver_stage(
                    source_output,
                    staging_dir,
                    "staged",
                    source_task_metadata,
                    0,
                )
                if is_cancelled():
                    raise Exception("任务已被宿主取消")
                progress("生成中", 45)
                upscale_metadata = dict(source_task_metadata)
                upscale_metadata.update({
                    "model": upscale_model,
                    "protocol": "openai_image" if upscale_model in GPT_IMAGE_MODELS else "gemini",
                    "pipeline_stage": "upscale",
                    "source_model": model,
                    "source_task_id": source_task_id,
                    "reference_image_count": 1,
                    "image_size": upscale_image_size,
                    "prompt_preview": _prompt_preview(_render_upscale_prompt(
                        upscale_prompt_template, upscale_image_size, aspect_ratio
                    )),
                })
                upscale_output = request_stage(
                    upscale_model,
                    _render_upscale_prompt(upscale_prompt_template, upscale_image_size, aspect_ratio),
                    {0: source_path},
                    _new_request_id(context, f"{position}_upscale"),
                    upscale_metadata,
                    45,
                    92,
                    upscale_image_size,
                )
                final_path, _ = deliver_stage(
                    upscale_output,
                    output_dir,
                    "delivered",
                    upscale_metadata,
                    position,
                    compress_result=True,
                )
        else:
            final_path, _ = deliver_stage(
                request_stage(
                    model,
                    prompt,
                    reference_images,
                    _new_request_id(context, position),
                    source_task_metadata,
                    8 + int(index * 70 / batch_num),
                    86,
                    image_size,
                    references_prepared=True,
                ),
                output_dir,
                "delivered",
                source_task_metadata,
                position,
                compress_result=True,
            )

        progress("完成", 86 + int((index + 1) * 14 / batch_num))
        return index, final_path

    def run_one(index):
        reference_slot = False
        task_slot = False
        progress("排队中", 5)
        try:
            if reference_image_count:
                _local_reference_gate.acquire(is_cancelled)
                reference_slot = True
            _local_task_gate.acquire(is_cancelled)
            task_slot = True
            return execute_one(index)
        finally:
            if task_slot:
                _local_task_gate.release()
            if reference_slot:
                _local_reference_gate.release()

    try:
        # Submit and deliver each batch item concurrently. Keep the returned
        # list deterministic so the host maps files back to the right slots.
        with ThreadPoolExecutor(
            max_workers=min(batch_num, _LOCAL_MAX_ACTIVE_TASKS),
            thread_name_prefix="yaliai-image",
        ) as executor:
            futures = [executor.submit(run_one, index) for index in range(batch_num)]
            errors = []
            for future in as_completed(futures):
                try:
                    index, path = future.result()
                    generated_files[index] = path
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]

        print(f"鸭梨 AI 异步任务完成，共保存 {len(generated_files)} 张图片")
        return generated_files
    except Exception as exc:
        print(f"鸭梨 AI 异步任务失败: {exc}")
        raise Exception(f"PLUGIN_ERROR:::{exc}") from exc
    finally:
        cleanup_reference_images()


# ===================== 初始化 =====================

_migrate_legacy_state()
_ensure_config_exists()

print("[Yali AI Image] 插件已加载")
print(f"[Yali AI Image] 状态目录: {_STATE_DIR}")
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
