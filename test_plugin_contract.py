import base64
import hashlib
import importlib.util
import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image


PLUGIN_PATH = Path(__file__).with_name("main.py")
_spec = importlib.util.spec_from_file_location("yaliai_image_plugin_test", PLUGIN_PATH)
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)


PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class _Session:
    def close(self):
        self.closed = True


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _GatewaySession:
    def __init__(self):
        self.post_calls = []
        self.get_calls = []
        self.poll_count = 0

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return _Response(202, {
            "task_id": "task-contract-1",
            "query_path": "/v1/image/tasks/task-contract-1",
            "status": "queued",
        })

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        self.poll_count += 1
        if self.poll_count == 1:
            return _Response(200, {"status": "queued"})
        return _Response(200, {
            "status": "completed",
            "data": [{"url": "https://api.yaliai.com/v1/generated-images/test.png"}],
        })


class _DownloadResponse:
    status_code = 200
    text = ""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, chunk_size):
        del chunk_size
        yield base64.b64decode(PNG_1X1)


class _UpdateDownloadResponse:
    def __init__(self, payload, url="https://api.yaliai.com/downloads/yaliai-async-image-plugin.zip"):
        self.status_code = 200
        self.url = url
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, chunk_size):
        for offset in range(0, len(self._payload), chunk_size):
            yield self._payload[offset:offset + chunk_size]


class _DownloadSession:
    def __init__(self):
        self.get_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return _DownloadResponse()


class PluginContractTests(unittest.TestCase):
    def test_host_state_uses_the_official_user_resources_location(self):
        original_dir = plugin.plugin_dir
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                app_dir = Path(temp_dir) / "TypeTale"
                plugin.plugin_dir = app_dir / "_internal" / "plugins" / "image_plugins" / "yaliai-async-image-plugin"
                (app_dir / "user_resources").mkdir(parents=True)
                self.assertEqual(
                    plugin._resolve_state_dir(),
                    app_dir / "user_resources" / "plugins" / "image_plugins" / "yaliai-async-image-plugin",
                )
        finally:
            plugin.plugin_dir = original_dir

    def test_ui_uses_plugin_sdk_for_persistence_without_reloading_stale_config(self):
        ui_html = (PLUGIN_PATH.parent / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn("PluginSDK.saveParam(key, value)", ui_html)
        self.assertNotIn("PluginSDK.sendAction('save_param'", ui_html)
        self.assertNotIn("PluginSDK.sendAction('load_params')", ui_html)
        self.assertIn("result.error !== '后端未返回插件动作结果", ui_html)
        for action in (
            "save_param",
            "get_configured_references",
            "save_configured_references",
            "clear_configured_references",
            "open_task_logs",
            "check_plugin_update",
            "install_plugin_update",
        ):
            self.assertIn(f"PluginSDK.onAction('{action}'", ui_html)
        self.assertIn('id="checkPluginUpdate"', ui_html)
        self.assertIn("PluginSDK.sendAction('check_plugin_update'", ui_html)
        self.assertIn("PluginSDK.sendAction('install_plugin_update'", ui_html)

    def test_handle_action_accepts_host_context_argument(self):
        original_params = dict(plugin._global_params)
        try:
            result = plugin.handle_action("save_param", {"key": "model", "value": "gpt-image2-Pro"}, {})
            self.assertTrue(result["ok"])
            self.assertEqual(plugin._global_params["model"], "gpt-image-2")
        finally:
            plugin._global_params.clear()
            plugin._global_params.update(original_params)

    def test_plugin_update_requires_a_strictly_newer_numeric_version(self):
        self.assertTrue(plugin._is_newer_version("3.4.2", "3.4.1"))
        self.assertFalse(plugin._is_newer_version("3.4.1", "3.4.1"))
        self.assertFalse(plugin._is_newer_version("3.4.0", "3.4.1"))
        with self.assertRaises(ValueError):
            plugin._is_newer_version("latest", "3.4.1")

    def test_plugin_update_requires_trusted_https_download_host(self):
        self.assertTrue(plugin._is_allowed_update_url(
            "https://api.yaliai.com/downloads/yaliai-async-image-plugin.zip"
        ))
        self.assertFalse(plugin._is_allowed_update_url(
            "http://api.yaliai.com/downloads/yaliai-async-image-plugin.zip"
        ))
        self.assertFalse(plugin._is_allowed_update_url(
            "https://example.com/yaliai-async-image-plugin.zip"
        ))

    def test_plugin_update_rejects_bad_checksum_before_download(self):
        result = plugin._install_plugin_update(
            "https://api.yaliai.com/downloads/yaliai-async-image-plugin.zip",
            "invalid",
            "3.4.2",
        )
        self.assertFalse(result["ok"])
        self.assertIn("sha256", result["error"])

    def test_plugin_update_rejects_install_while_generation_is_active(self):
        checksum = "0" * 64
        with plugin._generation_jobs_lock:
            plugin._generation_jobs.add("test-active-job")
        try:
            result = plugin._install_plugin_update(
                "https://api.yaliai.com/downloads/yaliai-async-image-plugin.zip",
                checksum,
                "3.4.2",
            )
        finally:
            with plugin._generation_jobs_lock:
                plugin._generation_jobs.discard("test-active-job")
        self.assertFalse(result["ok"])
        self.assertIn("正在执行", result["error"])

    def test_plugin_update_accepts_only_safe_complete_package_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package = Path(temp_dir) / "plugin.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("yaliai-async-image-plugin/main.py", '_PLUGIN_VERSION = "3.4.2"\n')
                archive.writestr("yaliai-async-image-plugin/ui/index.html", "<html></html>")
                archive.writestr("yaliai-async-image-plugin/ui/task_log.html", "<html></html>")
            extract_dir = Path(temp_dir) / "extract"
            extract_dir.mkdir()
            source = plugin._safe_extract_update_package(package, extract_dir)
            self.assertEqual(source.name, "yaliai-async-image-plugin")

    def test_plugin_update_rejects_path_traversal_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package = Path(temp_dir) / "unsafe.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("../main.py", "unsafe")
            extract_dir = Path(temp_dir) / "extract"
            extract_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "非法路径"):
                plugin._safe_extract_update_package(package, extract_dir)

    def test_plugin_update_installs_verified_package_and_preserves_user_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "yaliai-async-image-plugin"
            (target / "ui").mkdir(parents=True)
            (target / "main.py").write_text('_PLUGIN_VERSION = "3.4.1"\nold = True\n', encoding="utf-8")
            (target / "ui" / "index.html").write_text("old index", encoding="utf-8")
            (target / "ui" / "task_log.html").write_text("old task log", encoding="utf-8")
            (target / "config.json").write_text('{"gpt_api_key":"local-only"}', encoding="utf-8")

            package = root / "update.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("yaliai-async-image-plugin/main.py", '_PLUGIN_VERSION = "3.4.2"\nnew = True\n')
                archive.writestr("yaliai-async-image-plugin/ui/index.html", "new index")
                archive.writestr("yaliai-async-image-plugin/ui/task_log.html", "new task log")
                archive.writestr("yaliai-async-image-plugin/README.md", "new readme")
            package_bytes = package.read_bytes()
            checksum = hashlib.sha256(package_bytes).hexdigest()

            original_plugin_dir = plugin.plugin_dir
            try:
                plugin.plugin_dir = target
                with patch.object(plugin.requests, "get", return_value=_UpdateDownloadResponse(package_bytes)):
                    result = plugin._install_plugin_update(
                        "https://api.yaliai.com/downloads/yaliai-async-image-plugin.zip",
                        checksum,
                        "3.4.2",
                    )
            finally:
                plugin.plugin_dir = original_plugin_dir

            self.assertTrue(result["ok"])
            self.assertIn('_PLUGIN_VERSION = "3.4.2"', (target / "main.py").read_text(encoding="utf-8"))
            self.assertEqual((target / "ui" / "index.html").read_text(encoding="utf-8"), "new index")
            self.assertEqual((target / "config.json").read_text(encoding="utf-8"), '{"gpt_api_key":"local-only"}')
            backup = Path(result["backup_path"])
            self.assertEqual((backup / "main.py").read_text(encoding="utf-8"), '_PLUGIN_VERSION = "3.4.1"\nold = True\n')

    def test_update_check_reads_manifest_and_returns_signed_package_metadata(self):
        manifest = {
            "version": "3.4.2",
            "download_url": "https://api.yaliai.com/downloads/yaliai-async-image-plugin.zip",
            "sha256": hashlib.sha256(b"package").hexdigest(),
            "notes": "更新说明",
        }
        with patch.object(plugin.requests, "get", return_value=_Response(200, manifest)) as request:
            result = plugin._check_plugin_update()
        self.assertTrue(result["ok"])
        self.assertTrue(result["has_update"])
        self.assertEqual(result["remote_version"], "3.4.2")
        self.assertEqual(result["notes"], "更新说明")
        self.assertEqual(request.call_args.args[0], plugin._UPDATE_MANIFEST_URL)

    def test_manual_upscale_refreshes_the_host_selected_asset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "panel.png"
            source.write_bytes(base64.b64decode(PNG_1X1))
            calls = []

            def fake_post(url, **kwargs):
                calls.append((url, kwargs["json"]))
                tool = kwargs["json"]["name"]
                if tool == "zzdh_get_edit_view_data" and len(calls) == 1:
                    return _Response(200, {
                        "images": [{"path": str(source), "version": 10}],
                        "selected_index": 0,
                    })
                if tool == "zzdh_import_image_from_path":
                    return _Response(200, {"unique_name": "panel-a", "image_index": 0})
                if tool == "zzdh_get_edit_view_data":
                    return _Response(200, {
                        "images": [{"path": str(source), "version": 11}],
                        "selected_index": 0,
                    })
                self.fail(f"unexpected host tool call: {tool}")

            with patch.object(plugin.requests, "post", side_effect=fake_post):
                result = plugin._refresh_host_image_after_manual_upscale(source, "panel-a")

            self.assertEqual(result["state"], "refreshed")
            self.assertEqual(result["image_index"], 0)
            self.assertEqual(result["image_version"], 11)
            self.assertEqual([item[1]["name"] for item in calls], [
                "zzdh_get_edit_view_data", "zzdh_import_image_from_path", "zzdh_get_edit_view_data",
            ])
            self.assertEqual(calls[1][1]["arguments"]["image_path"], str(source.resolve()))

    def test_manual_upscale_storyboard_refresh_accepts_new_jpeg_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "panel.png"
            replacement = Path(temp_dir) / "panel.jpg"
            source.write_bytes(base64.b64decode(PNG_1X1))
            with Image.new("RGB", (1, 1), (0, 0, 0)) as image:
                image.save(replacement, "JPEG")
            responses = [
                {"images": [{"path": str(source), "version": 10}], "selected_index": 0},
                {"image_index": 0},
                {"images": [{"path": str(replacement), "version": 11}], "selected_index": 0},
            ]

            with patch.object(plugin, "_call_host_tool", side_effect=responses) as host_call:
                result = plugin._refresh_host_image_after_manual_upscale(
                    replacement, "panel-a", expected_source=source
                )

            self.assertEqual(result["state"], "refreshed")
            self.assertEqual(result["path"], str(replacement))
            self.assertEqual(host_call.call_args_list[1].args, ("zzdh_import_image_from_path", {
                "unique_name": "panel-a", "image_path": str(replacement.resolve()),
            }))

    def test_storyboard_probe_requires_exact_selected_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "panel.png"
            source.write_bytes(base64.b64decode(PNG_1X1))
            with patch.object(plugin, "_call_host_tool", return_value={
                "images": [{"path": str(source), "version": 10}], "selected_index": 0,
            }):
                result = plugin._probe_host_storyboard_source(source, "panel-a")
            self.assertTrue(result["is_storyboard"])

    def test_entity_or_unknown_manual_target_keeps_original_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "entity.png"
            source.write_bytes(base64.b64decode(PNG_1X1))
            with patch.object(plugin, "_call_host_tool", side_effect=RuntimeError("not a panel")):
                probe = plugin._probe_host_storyboard_source(source, "entity-1")
            self.assertFalse(probe["is_storyboard"])
            staged = plugin._compress_manual_replacement_output(
                source, source.suffix, {"local_result_max_mb": 1}
            )
            self.assertEqual(Path(staged).suffix, ".png")

    def test_manual_upscale_does_not_refresh_a_newer_host_asset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "panel.png"
            newer = Path(temp_dir) / "newer.png"
            source.write_bytes(base64.b64decode(PNG_1X1))
            newer.write_bytes(base64.b64decode(PNG_1X1))

            with patch.object(plugin, "_call_host_tool", return_value={
                "images": [{"path": str(newer), "version": 12}], "selected_index": 0,
            }) as host_call:
                result = plugin._refresh_host_image_after_manual_upscale(source, "panel-a")

            self.assertEqual(result["state"], "skipped")
            self.assertIn("已变化", result["reason"])
            host_call.assert_called_once_with("zzdh_get_edit_view_data", {"unique_name": "panel-a"})

    def test_manual_upscale_keeps_success_when_host_refresh_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "panel.png"
            source.write_bytes(base64.b64decode(PNG_1X1))

            with patch.object(plugin, "_call_host_tool", side_effect=RuntimeError("connection refused")):
                result = plugin._refresh_host_image_after_manual_upscale(source, "panel-a")

            self.assertEqual(result["state"], "unavailable")
            self.assertIn("connection refused", result["reason"])

    def test_delivered_task_can_be_explicitly_set_as_host_keyframe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "delivered.png"
            image_path.write_bytes(base64.b64decode(PNG_1X1))
            events = [{
                "task_id": "task-frame-1",
                "timestamp": 1,
                "status": "success",
                "unique_name": "panel-a",
                "output_path": str(image_path),
            }]
            with patch.object(plugin, "_read_async_task_events", return_value=events), \
                    patch.object(plugin, "_call_host_tool", return_value={"success": True}) as host_call, \
                    patch.object(plugin, "_record_async_task") as record:
                result = plugin._set_task_frame("task-frame-1", "first")

            self.assertTrue(result["ok"])
            self.assertEqual(result["message"], "已设为分镜首帧")
            host_call.assert_called_once_with("zzdh_set_first_frame", {
                "unique_name": "panel-a", "image_path": str(image_path.resolve()),
            })
            record.assert_called_once_with(
                "host_frame_linked", task_id="task-frame-1", status="success",
                host_frame="first", host_frame_path=str(image_path.resolve()),
            )

    def test_non_delivered_task_cannot_be_set_as_host_keyframe(self):
        events = [{
            "task_id": "task-frame-2",
            "timestamp": 1,
            "status": "processing",
            "unique_name": "panel-a",
            "output_path": "C:/missing.png",
        }]
        with patch.object(plugin, "_read_async_task_events", return_value=events), \
                patch.object(plugin, "_call_host_tool") as host_call:
            result = plugin._set_task_frame("task-frame-2", "end")

        self.assertFalse(result["ok"])
        self.assertIn("已交付", result["error"])
        host_call.assert_not_called()

    def test_configured_references_persist_and_clear_from_plugin_state(self):
        original_config_path = plugin._CONFIG_PATH
        original_reference_dir = plugin._CONFIGURED_REFERENCE_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                state_dir = Path(temp_dir) / "user_resources" / "plugins" / "image_plugins" / "plugin"
                plugin._CONFIG_PATH = state_dir / "config.json"
                plugin._CONFIGURED_REFERENCE_DIR = state_dir / "configured_references"
                data_url = "data:image/png;base64," + PNG_1X1
                saved = plugin.handle_action("save_configured_references", {
                    "images": [{"name": "reference.png", "data_url": data_url}],
                })
                self.assertTrue(saved["ok"])
                self.assertEqual(len(saved["images"]), 1)
                saved_path = Path(saved["images"][0]["path"])
                self.assertTrue(saved_path.is_file())
                self.assertTrue(saved_path.is_relative_to(plugin._CONFIGURED_REFERENCE_DIR))
                self.assertEqual(plugin._configured_reference_paths(), [str(saved_path)])

                cleared = plugin.handle_action("clear_configured_references")
                self.assertTrue(cleared["ok"])
                self.assertFalse(saved_path.exists())
                self.assertEqual(plugin._configured_reference_paths(), [])
            finally:
                plugin._CONFIG_PATH = original_config_path
                plugin._CONFIGURED_REFERENCE_DIR = original_reference_dir

    def test_reference_normalization_supports_host_preset_shapes(self):
        with tempfile.TemporaryDirectory() as source_dir:
            first = str(Path(source_dir) / "first.png")
            second = str(Path(source_dir) / "second.png")
            third = str(Path(source_dir) / "third.png")
            references = {
                "参考图片MAP": {
                    "1": {"path": second},
                    "0": first,
                },
                "首帧": {"image_path": first},
                "尾帧": {"file_path": third},
            }

            normalized = plugin._normalize_reference_images(references)

        self.assertEqual(list(normalized.values()), [first, second, third])

    def test_reference_normalization_keeps_mixed_top_level_order(self):
        references = {
            "首帧": "first.png",
            "参考图片MAP": {"1": "map-second.png", "0": "map-first.png"},
            "尾帧": {"path": "last.png"},
        }

        normalized = plugin._normalize_reference_images(references)

        self.assertEqual(
            list(normalized.values()),
            ["first.png", "map-first.png", "map-second.png", "last.png"],
        )

    def test_gemini_reference_parts_keep_reference_order(self):
        with tempfile.TemporaryDirectory() as source_dir:
            first = Path(source_dir) / "first.png"
            second = Path(source_dir) / "second.png"
            first.write_bytes(base64.b64decode(PNG_1X1))
            second.write_bytes(base64.b64decode(PNG_1X1))

            parts = plugin.build_gemini_parts(
                "prompt",
                {0: str(first), 1: str(second)},
            )

        self.assertEqual([part["text"] for part in parts[:1]], ["prompt"])
        self.assertEqual(
            [part["inlineData"]["data"] for part in parts[1:]],
            [PNG_1X1, PNG_1X1],
        )

    def test_openai_multipart_reference_fields_keep_reference_order(self):
        submitted = {}

        def capture_submit(_session, _url, _api_key, _request_id, _timeout, **kwargs):
            submitted.update(kwargs)
            return {"task_id": "ordered-task", "query_path": "/v1/image/tasks/ordered-task"}

        with tempfile.TemporaryDirectory() as source_dir:
            first = Path(source_dir) / "first.png"
            second = Path(source_dir) / "second.png"
            first.write_bytes(base64.b64decode(PNG_1X1))
            second.write_bytes(base64.b64decode(PNG_1X1))

            with patch.object(plugin, "_submit_async_request", side_effect=capture_submit), \
                    patch.object(plugin, "_poll_async_task", return_value=[]):
                plugin.send_gpt_image_request(
                    api_key="test-key",
                    endpoint="http://gateway.invalid",
                    model="gpt-image-2",
                    prompt="prompt",
                    reference_images={0: str(first), 1: str(second)},
                    session=_Session(),
                )

        self.assertEqual(
            [item[1][0] for item in submitted["files"]],
            ["first.png", "second.png"],
        )
        self.assertEqual(submitted["form_data"]["output_format"], "jpeg")

    def test_openai_generation_json_requests_jpeg_output(self):
        submitted = {}

        def capture_submit(_session, _url, _api_key, _request_id, _timeout, **kwargs):
            submitted.update(kwargs)
            return {"task_id": "jpeg-task", "query_path": "/v1/image/tasks/jpeg-task"}

        with patch.object(plugin, "_submit_async_request", side_effect=capture_submit), \
                patch.object(plugin, "_poll_async_task", return_value=[]):
            plugin.send_gpt_image_request(
                api_key="test-key",
                endpoint="http://gateway.invalid",
                model="gpt-image-2",
                prompt="prompt",
                reference_images={},
                session=_Session(),
            )

        self.assertEqual(submitted["json_payload"]["output_format"], "jpeg")
        self.assertEqual(submitted["json_payload"]["response_format"], "url")

    def test_reference_images_are_compressed_to_temporary_jpeg(self):
        with tempfile.TemporaryDirectory() as source_dir:
            source_path = Path(source_dir) / "reference.png"
            with Image.new("RGBA", (5000, 3000), (20, 80, 140, 255)) as image:
                image.save(source_path, "PNG")
            source_path.write_bytes(source_path.read_bytes() + b"x" * (6 * 1024 * 1024))
            original_size = source_path.stat().st_size

            prepared, cleanup = plugin._prepare_reference_images(
                {0: str(source_path)}, {}
            )
            try:
                prepared_path = Path(prepared[0])
                self.assertNotEqual(prepared_path, source_path)
                self.assertEqual(prepared_path.suffix, ".jpg")
                self.assertLess(prepared_path.stat().st_size, 5 * 1024 * 1024)
                self.assertEqual(source_path.stat().st_size, original_size)
                with Image.open(prepared_path) as image:
                    self.assertEqual(image.size, (4096, 2458))
            finally:
                cleanup()
            self.assertFalse(prepared_path.exists())

    def test_reference_images_at_or_below_five_mb_are_not_reencoded(self):
        with tempfile.TemporaryDirectory() as source_dir:
            source_path = Path(source_dir) / "small.png"
            source_path.write_bytes(base64.b64decode(PNG_1X1))
            prepared, cleanup = plugin._prepare_reference_images({0: str(source_path)}, {})
            try:
                self.assertEqual(prepared[0], str(source_path))
            finally:
                cleanup()

    def test_delivered_result_replaces_host_file_without_changing_png_path(self):
        with tempfile.TemporaryDirectory() as source_dir:
            source_path = Path(source_dir) / "storyboard.png"
            with Image.new("RGBA", (5000, 3000), (20, 80, 140, 255)) as image:
                image.save(source_path, "PNG")
            source_path.write_bytes(source_path.read_bytes() + b"x" * (2 * 1024 * 1024))

            result_path = plugin._compress_delivered_output(
                source_path,
                {"local_result_max_mb": 1},
            )

            self.assertEqual(result_path, str(source_path))
            self.assertTrue(source_path.exists())
            self.assertLessEqual(source_path.stat().st_size, 1 * 1024 * 1024)
            self.assertFalse(source_path.with_suffix(".jpg").exists())
            with Image.open(source_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (4096, 2458))

    def test_delivered_result_preserves_jpeg_and_webp_formats(self):
        for extension, image_format in ((".jpg", "JPEG"), (".webp", "WEBP")):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as source_dir:
                source_path = Path(source_dir) / f"generated{extension}"
                with Image.new("RGB", (2400, 1600), (120, 80, 40)) as image:
                    image.save(source_path, image_format)
                source_path.write_bytes(source_path.read_bytes() + b"x" * (2 * 1024 * 1024))

                result_path = plugin._compress_delivered_output(
                    source_path,
                    {"local_result_max_mb": 1},
                )

                self.assertEqual(result_path, str(source_path))
                with Image.open(source_path) as image:
                    self.assertEqual(image.format, "JPEG" if extension == ".jpg" else "WEBP")

    def test_fresh_delivered_png_can_become_jpeg_without_resizing(self):
        with tempfile.TemporaryDirectory() as source_dir:
            source_path = Path(source_dir) / "fresh.png"
            with Image.new("RGB", (4800, 3584), (120, 80, 40)) as image:
                image.save(source_path, "PNG")
            source_path.write_bytes(source_path.read_bytes() + b"x" * (2 * 1024 * 1024))

            result_path = plugin._compress_delivered_output(
                source_path,
                {"local_result_max_mb": 1},
                allow_format_change=True,
            )

            result = Path(result_path)
            self.assertEqual(result.suffix, ".jpg")
            self.assertFalse(source_path.exists())
            self.assertLessEqual(result.stat().st_size, 1 * 1024 * 1024)
            with Image.open(result) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (4800, 3584))

    def test_large_reference_is_compressed_instead_of_rejected(self):
        with tempfile.TemporaryDirectory() as source_dir:
            source_path = Path(source_dir) / "large.png"
            source_path.write_bytes(base64.b64decode(PNG_1X1) + b"x" * (11 * 1024 * 1024))
            prepared, cleanup = plugin._prepare_reference_images(
                {0: str(source_path)}, {"reference_image_target_mb": 2}
            )
            try:
                self.assertTrue(Path(prepared[0]).suffix == ".jpg")
                self.assertLessEqual(Path(prepared[0]).stat().st_size, 2 * 1024 * 1024)
            finally:
                cleanup()

    def test_default_openai_size_is_4k_16_by_9(self):
        self.assertEqual(plugin._default_params["quality"], "medium")
        self.assertEqual(plugin._default_params["image_size"], "4K")
        self.assertEqual(plugin._default_params["aspect_ratio"], "16:9")
        self.assertEqual(plugin.build_gpt_image_size("16:9", "4K"), "3840x2160")

    def test_async_polling_uses_a_short_bounded_interval(self):
        self.assertEqual(plugin._ASYNC_INITIAL_DELAY_SECONDS, 20)
        self.assertEqual(plugin._ASYNC_POLL_INTERVAL_SECONDS, 2)

    def test_upscale_prompt_replaces_size_and_ratio_variables(self):
        prompt = plugin._render_upscale_prompt(
            "target {{image_size}} / {{aspect_ratio}} and {image_size} / {aspect_ratio}",
            "4K",
            "16:9",
        )
        self.assertEqual(prompt, "target 4K / 16:9 and 4K / 16:9")

    def test_upscale_defaults_to_gemini_pro_and_doubles_workflow_budget(self):
        self.assertEqual(plugin._default_params["upscale_model"], "gemini-3-pro-image-preview")
        self.assertEqual(plugin._ASYNC_MAX_WAIT_SECONDS * 2, 3600)

    def test_model_family_uses_its_own_credential(self):
        params = {
            "gpt_api_key": "gpt-key",
            "gemini_api_key": "gemini-key",
            "api_key": "legacy-key",
        }
        self.assertEqual(plugin._api_key_for_model(params, "gpt-image-2"), "gpt-key")
        self.assertEqual(plugin._api_key_for_model(params, "gemini-3.1-flash-image-preview"), "gemini-key")

    def test_generation_uses_the_configured_gateway_endpoint(self):
        with tempfile.TemporaryDirectory() as output_dir, \
                patch.object(plugin, "send_gemini_request", return_value=[{"type": "b64", "value": PNG_1X1}]) as submit:
            plugin.generate({
                "prompt": "configured endpoint test",
                "output_dir": output_dir,
                "plugin_params": {
                    "gemini_api_key": "test-key",
                    "endpoint": "https://gateway.example.test/",
                    "model": "gemini-3.1-flash-image-preview",
                },
            })
        self.assertEqual(submit.call_args.kwargs["endpoint"], "https://gateway.example.test")
        self.assertEqual(plugin._normalize_endpoint(None), "https://api.yaliai.com")
        with self.assertRaisesRegex(Exception, "API URL"):
            plugin._normalize_endpoint("ftp://gateway.example.test")

    def test_upscale_mode_runs_source_then_upscale_and_returns_only_final_file(self):
        stage_order = []
        upscale_reference_paths = []
        final_file_exists = False

        def fake_source(**_kwargs):
            stage_order.append("source")
            return [{"type": "b64", "value": PNG_1X1, "task_id": "source-task"}]

        def fake_upscale(**kwargs):
            stage_order.append("upscale")
            reference_path = kwargs["reference_images"][0]
            upscale_reference_paths.append(reference_path)
            self.assertEqual(kwargs["model"], "gpt-image-2")
            self.assertEqual(kwargs["api_key"], "upscale-key")
            self.assertTrue(Path(reference_path).exists())
            self.assertIn("4K", kwargs["prompt"])
            self.assertIn("16:9", kwargs["prompt"])
            self.assertNotIn("{{image_size}}", kwargs["prompt"])
            return [{"type": "b64", "value": PNG_1X1, "task_id": "upscale-task"}]

        with tempfile.TemporaryDirectory() as output_dir, \
                patch.object(plugin, "send_gemini_request", side_effect=fake_source), \
                patch.object(plugin, "send_gpt_image_request", side_effect=fake_upscale), \
                patch.object(plugin, "_record_async_task"):
            paths = plugin.generate({
                "prompt": "source prompt",
                "output_dir": output_dir,
                "output_position": [3],
                "plugin_params": {
                    "gemini_api_key": "source-key",
                    "gpt_api_key": "upscale-key",
                    "model": "gemini-3.1-flash-image-preview",
                    "generation_mode": "upscale",
                    "upscale_model": "gpt-image-2",
                    "upscale_prompt": "enhance {{image_size}} {{aspect_ratio}}",
                    "image_size": "4K",
                    "aspect_ratio": "16:9",
                    "quality": "medium",
                },
            })
            final_file_exists = Path(paths[0]).exists()

        self.assertEqual(stage_order, ["source", "upscale"])
        self.assertEqual(len(upscale_reference_paths), 1)
        self.assertEqual(len(paths), 1)
        self.assertTrue(final_file_exists)
        self.assertFalse(Path(upscale_reference_paths[0]).exists())

    def test_upscale_gemini_model_uses_gemini_branch_and_key(self):
        calls = []

        def fake_gemini(**kwargs):
            calls.append((kwargs["model"], kwargs["api_key"], kwargs["reference_images"]))
            return [{"type": "b64", "value": PNG_1X1, "task_id": kwargs["model"]}]

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            plugin, "send_gemini_request", side_effect=fake_gemini
        ):
            plugin.generate({
                "prompt": "gemini upscale test",
                "output_dir": output_dir,
                "plugin_params": {
                    "gemini_api_key": "source-and-upscale-key",
                    "model": "gemini-3.1-flash-image-preview",
                    "generation_mode": "upscale",
                    "upscale_model": "gemini-3-pro-image-preview",
                    "upscale_prompt": "enhance {{image_size}} {{aspect_ratio}}",
                },
            })

        self.assertEqual([item[0] for item in calls], [
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
        ])
        self.assertEqual([item[1] for item in calls], [
            "source-and-upscale-key",
            "source-and-upscale-key",
        ])
        self.assertEqual(calls[1][2].keys(), {0})

    def test_batch_upscale_keeps_each_storyboard_stage_order_and_output_order(self):
        events = []
        events_lock = threading.Lock()

        def fake_gemini(**kwargs):
            stage = "upscale" if kwargs["reference_images"] else "source"
            with events_lock:
                events.append((stage, kwargs["request_id"]))
            return [{"type": "b64", "value": PNG_1X1, "task_id": kwargs["request_id"]}]

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            plugin, "send_gemini_request", side_effect=fake_gemini
        ):
            paths = plugin.generate({
                "prompt": "batch upscale ordering test",
                "output_dir": output_dir,
                "output_position": [9, 2, 7],
                "batch_num": 3,
                "plugin_params": {
                    "gemini_api_key": "test-key",
                    "model": "gemini-3.1-flash-image-preview",
                    "generation_mode": "upscale",
                    "upscale_model": "gemini-3-pro-image-preview",
                    "upscale_prompt": "enhance {{image_size}} {{aspect_ratio}}",
                },
            })

        self.assertEqual(len(paths), 3)
        self.assertTrue(Path(paths[0]).name.endswith("_9.png"))
        self.assertTrue(Path(paths[1]).name.endswith("_2.png"))
        self.assertTrue(Path(paths[2]).name.endswith("_7.png"))
        for position in (9, 2, 7):
            source_index = next(
                index for index, (stage, request_id) in enumerate(events)
                if stage == "source" and f"_0_{position}_" in request_id
            )
            upscale_index = next(
                index for index, (stage, request_id) in enumerate(events)
                if stage == "upscale" and f"_{position}_upscale_" in request_id
            )
            self.assertLess(source_index, upscale_index)

    def test_upscale_cross_protocol_keeps_semantic_size_for_gemini(self):
        source_calls = []
        upscale_calls = []

        def fake_gpt(**kwargs):
            source_calls.append(kwargs)
            return [{"type": "b64", "value": PNG_1X1, "task_id": "source-task"}]

        def fake_gemini(**kwargs):
            upscale_calls.append(kwargs)
            return [{"type": "b64", "value": PNG_1X1, "task_id": "upscale-task"}]

        with tempfile.TemporaryDirectory() as output_dir, \
                patch.object(plugin, "send_gpt_image_request", side_effect=fake_gpt), \
                patch.object(plugin, "send_gemini_request", side_effect=fake_gemini):
            plugin.generate({
                "prompt": "cross protocol upscale test",
                "output_dir": output_dir,
                "plugin_params": {
                    "gpt_api_key": "source-key",
                    "gemini_api_key": "upscale-key",
                    "model": "gpt-image-2",
                    "generation_mode": "upscale",
                    "upscale_model": "gemini-3-pro-image-preview",
                    "upscale_prompt": "enhance {{image_size}} {{aspect_ratio}}",
                    "image_size": "4K",
                    "aspect_ratio": "16:9",
                },
            })

        self.assertEqual(source_calls[0]["image_size"], "4K")
        self.assertEqual(source_calls[0]["aspect_ratio"], "16:9")
        self.assertEqual(upscale_calls[0]["image_size"], "4K")
        self.assertEqual(upscale_calls[0]["aspect_ratio"], "16:9")
        self.assertNotEqual(upscale_calls[0]["image_size"], "3840x2160")
        self.assertEqual(upscale_calls[0]["api_key"], "upscale-key")

    def test_upscale_mode_validates_second_model_key_before_source_request(self):
        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            plugin, "send_gemini_request"
        ) as source_request:
            with self.assertRaisesRegex(Exception, "超分模型所需"):
                plugin.generate({
                    "prompt": "missing upscale key",
                    "output_dir": output_dir,
                    "plugin_params": {
                        "gemini_api_key": "source-key",
                        "model": "gemini-3.1-flash-image-preview",
                        "generation_mode": "upscale",
                        "upscale_model": "gpt-image-2",
                    },
                })
        source_request.assert_not_called()

    def test_legacy_gpt_model_alias_uses_openai_images_branch(self):
        self.assertEqual(plugin._normalize_model("gpt-image2-Pro"), "gpt-image-2")
        self.assertEqual(plugin._normalize_model("openai-image-2"), "gpt-image-2")
        self.assertEqual(plugin._normalize_model("gpt-image-2"), "gpt-image-2")

        session = _Session()
        with tempfile.TemporaryDirectory() as output_dir, \
                patch.object(plugin, "_new_http_session", return_value=session), \
                patch.object(plugin, "send_gpt_image_request", return_value=[{"type": "b64", "value": PNG_1X1}]) as send_gpt, \
                patch.object(plugin, "send_gemini_request") as send_gemini:
            plugin.generate({
                "prompt": "alias branch test",
                "output_dir": output_dir,
                "plugin_params": {
                    "model": "gpt-image2-Pro",
                    "gpt_api_key": "gpt-key",
                    "gemini_api_key": "gemini-key",
                    "endpoint": "http://gateway.invalid",
                    "aspect_ratio": "16:9",
                    "image_size": "4K",
                    "quality": "medium",
                },
            })
        send_gpt.assert_called_once()
        self.assertEqual(send_gpt.call_args.kwargs["model"], "gpt-image-2")
        self.assertEqual(send_gpt.call_args.kwargs["api_key"], "gpt-key")
        send_gemini.assert_not_called()

    def test_task_log_popup_groups_events_and_can_clear(self):
        task_log_html = (PLUGIN_PATH.parent / "ui" / "task_log.html").read_text(encoding="utf-8")
        self.assertNotIn("分镜 / 输出", task_log_html)
        self.assertIn("var mode = task.generation_mode === 'upscale'", task_log_html)
        self.assertIn('.model { width: 175px;', task_log_html)
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            plugin, "_ASYNC_TASK_LOG_PATH", Path(temp_dir) / "async_tasks.jsonl"
        ):
            plugin._record_async_task(
                "accepted", task_id="task-1", status="queued", model="gpt-image-2",
                viewer_index=4, output_position=2, image_size="4K", request_size="3840x2160",
                aspect_ratio="16:9", quality="medium", prompt_preview="task preview",
            )
            plugin._record_async_task("status", task_id="task-1", status="processing")
            plugin._record_async_task("completed", task_id="task-1", status="completed", output_count=1)
            plugin._record_async_task(
                "delivered", task_id="task-1", status="success", output_path="C:/outputs/task-1.png",
                output_url="https://api.yaliai.com/v1/generated-images/task-1.png",
            )

            page = plugin.handle_action("get_task_logs", {"page": 1, "page_size": 10})
            self.assertTrue(page["ok"])
            self.assertEqual(page["total"], 1)
            self.assertEqual(page["tasks"][0]["event_count"], 4)
            self.assertEqual(page["tasks"][0]["viewer_index"], 4)
            self.assertEqual(page["tasks"][0]["request_size"], "3840x2160")
            self.assertEqual(page["tasks"][0]["status"], "success")
            self.assertEqual(plugin.handle_action("open_task_logs")["open_page"], "task_log.html")

            cleared = plugin.handle_action("clear_task_logs", {"mode": "all"})
            self.assertTrue(cleared["ok"])
            self.assertEqual(plugin.handle_action("get_task_logs")["total"], 0)

    def test_entity_manual_upscale_resolves_and_replaces_primary_and_thumbnail(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            plugin, "_ASYNC_TASK_LOG_PATH", Path(temp_dir) / "async_tasks.jsonl"
        ):
            image_path = Path(temp_dir) / "character_images" / "character_3.png"
            thumbnail_path = Path(temp_dir) / "thumbnails_500" / "character_3.jpg"
            image_path.parent.mkdir()
            thumbnail_path.parent.mkdir()
            Image.new("RGB", (8, 8), "white").save(image_path)
            Image.new("RGB", (4, 4), "white").save(thumbnail_path, "JPEG")
            plugin._record_async_task(
                "delivered", task_id="character-task", status="success",
                target_kind="character", target_id="3", unique_name="character_3",
                output_path=str(image_path),
            )
            summary = plugin._find_task_log_summary("character-task")
            host_entity = {
                "entities": [{
                    "character_id": "3",
                    "image_path": str(image_path),
                    "thumbnail_path": str(thumbnail_path),
                    "reference_items": [{"path": str(image_path), "media_type": "image"}],
                }],
            }
            with patch.object(plugin, "_call_host_tool", return_value=host_entity):
                asset = plugin._entity_asset_from_host(summary)
            self.assertEqual(asset["image_path"], str(image_path.resolve()))
            self.assertEqual(asset["thumbnail_path"], str(thumbnail_path.resolve()))

            staged = Path(temp_dir) / "staged.png"
            Image.new("RGB", (16, 12), "red").save(staged)
            original_hash = plugin._file_fingerprint(image_path)
            replacement = plugin._replace_entity_asset(str(staged), asset, original_hash)
            self.assertEqual(replacement["image_path"], str(image_path.resolve()))
            self.assertEqual(replacement["thumbnail_path"], str(thumbnail_path.resolve()))
            with Image.open(image_path) as image:
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            with Image.open(thumbnail_path) as thumbnail:
                self.assertEqual(thumbnail.format, "JPEG")
            self.assertTrue(Path(replacement["backup_path"]).is_file())
            self.assertTrue(Path(replacement["thumbnail_backup_path"]).is_file())

    def test_entity_asset_resolution_uses_one_common_path_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for kind, entity_id, suffix in (("character", "3", ".png"), ("location", "2", ".jpg"), ("item", "1", ".webp")):
                with self.subTest(kind=kind):
                    image_path = Path(temp_dir) / f"{kind}_{entity_id}{suffix}"
                    thumbnail_path = Path(temp_dir) / f"thumb_{kind}_{entity_id}.jpg"
                    Image.new("RGB", (4, 4), "white").save(image_path)
                    Image.new("RGB", (2, 2), "white").save(thumbnail_path, "JPEG")
                    id_field = f"{kind}_id"
                    summary = {"target_kind": kind, "target_id": entity_id, "target_name": ""}
                    payload = {"entities": [{
                        id_field: entity_id,
                        "image_path": str(image_path),
                        "thumbnail_path": str(thumbnail_path),
                    }]}
                    with patch.object(plugin, "_call_host_tool", return_value=payload) as host_call:
                        asset = plugin._entity_asset_from_host(summary)
                    self.assertEqual(asset["image_path"], str(image_path.resolve()))
                    self.assertEqual(asset["thumbnail_path"], str(thumbnail_path.resolve()))
                    host_call.assert_called_once_with("zzdh_get_entity_list", {"entity_type": kind})

    def test_task_log_popup_retries_when_host_bridge_is_not_ready(self):
        task_log_html = (PLUGIN_PATH.parent / "ui" / "task_log.html").read_text(encoding="utf-8")
        self.assertIn("window.parent && window.parent !== window", task_log_html)
        self.assertIn("window.opener && !window.opener.closed", task_log_html)
        self.assertIn("return targets.length > 0", task_log_html)
        self.assertIn("logLoadAttempts < 10", task_log_html)
        self.assertIn("}, 800)", task_log_html)
        self.assertIn("window.addEventListener('focus'", task_log_html)
        self.assertIn("clearTimeout(logLoadRetryTimer)", task_log_html)

    def test_task_log_does_not_open_local_images_from_popup(self):
        task_log_html = (PLUGIN_PATH.parent / "ui" / "task_log.html").read_text(encoding="utf-8")
        self.assertIn("important-notice", task_log_html)
        self.assertIn("避免在此页面操作超分导致软件界面无法查看最新效果", task_log_html)
        self.assertIn("local-copy-btn", task_log_html)
        self.assertIn("data-copy-local-path", task_log_html)
        self.assertNotIn("open_local_task_image", task_log_html)
        self.assertNotIn("正在打开本地图片", task_log_html)

    def test_generation_target_metadata_marks_entity_entry_points(self):
        self.assertEqual(
            plugin._generation_target_metadata({"unique_name": "character_2", "viewer_index": 0}),
            {"target_kind": "character", "target_id": "2", "target_name": ""},
        )
        self.assertEqual(
            plugin._generation_target_metadata({"unique_name": "location_1", "viewer_index": 0}),
            {"target_kind": "location", "target_id": "1", "target_name": ""},
        )
        self.assertEqual(
            plugin._generation_target_metadata({"unique_name": "item_1", "viewer_index": 0}),
            {"target_kind": "item", "target_id": "1", "target_name": ""},
        )
        self.assertEqual(
            plugin._generation_target_metadata({
                "entity_type": "location",
                "entity_id": "7",
                "entity_name": "城堡",
                "unique_name": "arbitrary-host-id",
            }),
            {"target_kind": "location", "target_id": "7", "target_name": "城堡"},
        )
        self.assertEqual(
            plugin._generation_target_metadata({"unique_name": "shot-a", "viewer_index": 3})["target_kind"],
            "storyboard",
        )
        self.assertEqual(
            plugin._generation_target_metadata({"unique_name": "standalone"})["target_kind"],
            "unknown",
        )
        old_log_summary = plugin._summarize_async_task_events([
            {"timestamp": 1, "event": "delivered", "task_id": "old-character", "status": "success",
             "unique_name": "character_9", "viewer_index": 0},
        ])[0]
        self.assertEqual(old_log_summary["target_kind"], "character")
        self.assertEqual(old_log_summary["target_id"], "9")

        task_log_html = (PLUGIN_PATH.parent / "ui" / "task_log.html").read_text(encoding="utf-8")
        self.assertIn("target_kind", task_log_html)
        self.assertIn("targetReference", task_log_html)

    def test_ui_uses_settings_panel_and_mode_buttons(self):
        ui_html = (PLUGIN_PATH.parent / "ui" / "index.html").read_text(encoding="utf-8")
        self.assertIn('<label class="form-label">API URL</label>', ui_html)
        self.assertIn('id="settingsPanel"', ui_html)
        self.assertIn('id="openSettings"', ui_html)
        self.assertIn('id="endpointInput"', ui_html)
        self.assertIn('.settings-backdrop[hidden] { display: none; }', ui_html)
        self.assertIn('.form-row[hidden] { display: none; }', ui_html)
        self.assertIn('class="mode-btn active"', ui_html)
        self.assertIn('data-mode="upscale"', ui_html)
        self.assertIn("upscale_model: 'gemini-3-pro-image-preview'", ui_html)
        self.assertIn('>超分</button>', ui_html)
        self.assertIn('>超分模型</label>', ui_html)
        self.assertIn('先使用默认模型生成基础图片，再使用超分模型生成最终图片', ui_html)
        self.assertIn('可使用变量：{image_size}（图像档位）和 {aspect_ratio}（图像比例）', ui_html)
        self.assertIn('id="upscaleHint" hidden', ui_html)
        self.assertLess(ui_html.index('生成模式'), ui_html.index('id="settingsPanel"'))
        self.assertNotIn('id="generationModeSelect"', ui_html)

    def test_upstream_image_download_allows_invalid_tls(self):
        session = _DownloadSession()
        with tempfile.TemporaryDirectory() as output_dir:
            output = plugin.download_url_to_output(
                "https://expired.example.test/image.png",
                {"viewer_index": 1, "unique_name": "tls", "generation_round": 0},
                output_dir,
                session=session,
            )

        self.assertEqual(session.get_calls[0][1]["verify"], False)
        self.assertTrue(Path(output).name.endswith("_0.png"))

    def test_async_gateway_submit_and_poll_contract(self):
        session = _GatewaySession()
        request_id = "yaliai_plugin_contract_unique"
        progress_texts = []
        with patch.object(plugin, "_sleep_with_cancel", side_effect=lambda *_: None), \
                patch.object(plugin, "_record_async_task"):
            accepted = plugin._submit_async_request(
                session,
                "https://api.yaliai.com/v1/images/generations",
                "test-key",
                request_id,
                5,
                json_payload={"async": True, "model": "gpt-image-2", "n": 1},
            )
            outputs = plugin._poll_async_task(
                session,
                "https://api.yaliai.com",
                "test-key",
                accepted,
                5,
                30,
                5,
                60,
                lambda text, *_: progress_texts.append(text),
            )

        self.assertEqual(accepted["task_id"], "task-contract-1")
        self.assertEqual(outputs, [{"type": "url", "value": "https://api.yaliai.com/v1/generated-images/test.png", "task_id": "task-contract-1"}])
        headers = session.post_calls[0][1]["headers"]
        self.assertEqual(headers["Idempotency-Key"], request_id)
        self.assertEqual(headers["X-Request-ID"], request_id)
        self.assertEqual(len(session.get_calls), 2)
        self.assertEqual(progress_texts, ["已提交", "排队中", "下载中"])
        self.assertTrue(all(len(text) <= 3 for text in progress_texts))

    def test_batch_submits_in_parallel_and_preserves_positions(self):
        request_ids = []
        session = _Session()
        barrier = threading.Barrier(2, timeout=3)

        def fake_gemini(**kwargs):
            request_ids.append(kwargs["request_id"])
            barrier.wait()
            return [{"type": "b64", "value": PNG_1X1}]

        with tempfile.TemporaryDirectory() as output_dir, \
                patch.object(plugin, "_new_http_session", return_value=session), \
                patch.object(plugin, "send_gemini_request", side_effect=fake_gemini):
            paths = plugin.generate({
                "prompt": "contract test",
                "reference_images": {},
                "output_dir": output_dir,
                "viewer_index": 2,
                "unique_name": "shot-a",
                "generation_round": 3,
                "output_position": [4, 1],
                "batch_num": 2,
                "plugin_params": {
                    "gemini_api_key": "test-key",
                    "endpoint": "http://gateway.invalid",
                    "model": "gemini-3.1-flash-image-preview",
                    "aspect_ratio": "16:9",
                    "image_size": "4K",
                    "quality": "medium",
                },
            })

        self.assertEqual(
            [Path(path).name for path in paths],
            ["0002_shot-a_3_4.png", "0002_shot-a_3_1.png"],
        )
        self.assertEqual(len(request_ids), 2)
        self.assertEqual(len(set(request_ids)), 2)
        self.assertTrue(all(f"_{position}_" in request_id for request_id, position in zip(request_ids, [4, 1])))

    def test_reference_batch_limits_local_uploads_to_twenty_tasks(self):
        active = 0
        peak_active = 0
        active_lock = threading.Lock()

        def fake_gemini(**_kwargs):
            nonlocal active, peak_active
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                time.sleep(0.05)
                return [{"type": "b64", "value": PNG_1X1}]
            finally:
                with active_lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            plugin, "send_gemini_request", side_effect=fake_gemini
        ):
            paths = plugin.generate({
                "prompt": "reference concurrency test",
                "reference_images": {"first": "C:/reference.png"},
                "output_dir": output_dir,
                "output_position": list(range(25)),
                "batch_num": 25,
                "plugin_params": {
                    "gemini_api_key": "test-key",
                    "endpoint": "http://gateway.invalid",
                    "model": "gemini-3.1-flash-image-preview",
                },
            })

        self.assertEqual(len(paths), 25)
        self.assertEqual(peak_active, 20)

    def test_global_gate_limits_multiple_host_calls_to_forty_tasks(self):
        active = 0
        peak_active = 0
        active_lock = threading.Lock()
        errors = []

        def fake_gemini(**_kwargs):
            nonlocal active, peak_active
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                time.sleep(0.05)
                return [{"type": "b64", "value": PNG_1X1}]
            finally:
                with active_lock:
                    active -= 1

        def call_plugin(output_dir, unique_name):
            try:
                plugin.generate({
                    "prompt": "global queue test",
                    "output_dir": output_dir,
                    "unique_name": unique_name,
                    "output_position": list(range(25)),
                    "batch_num": 25,
                    "plugin_params": {
                        "gemini_api_key": "test-key",
                        "endpoint": "http://gateway.invalid",
                        "model": "gemini-3.1-flash-image-preview",
                    },
                })
            except Exception as error:
                errors.append(error)

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            plugin, "send_gemini_request", side_effect=fake_gemini
        ):
            first = threading.Thread(target=call_plugin, args=(str(Path(output_dir) / "first"), "first"))
            second = threading.Thread(target=call_plugin, args=(str(Path(output_dir) / "second"), "second"))
            first.start()
            second.start()
            first.join(timeout=10)
            second.join(timeout=10)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(peak_active, 40)

    def test_cancelled_host_does_not_submit(self):
        session = _Session()
        with tempfile.TemporaryDirectory() as output_dir, \
                patch.object(plugin, "_new_http_session", return_value=session), \
                patch.object(plugin, "send_gemini_request") as submit:
            with self.assertRaisesRegex(Exception, "任务已被宿主取消"):
                plugin.generate({
                    "prompt": "cancel test",
                    "output_dir": output_dir,
                    "cancelled": True,
                    "plugin_params": {"api_key": "test-key", "endpoint": "http://gateway.invalid"},
                })
        submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
