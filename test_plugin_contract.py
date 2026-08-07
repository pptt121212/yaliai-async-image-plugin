import base64
import importlib.util
import tempfile
import threading
import time
import unittest
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


class _DownloadSession:
    def __init__(self):
        self.get_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return _DownloadResponse()


class PluginContractTests(unittest.TestCase):
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
        self.assertIn("generation_mode === 'upscale' ? '超分' : '默认'", task_log_html)
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
