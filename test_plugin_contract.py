import base64
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
    def test_default_openai_size_is_4k_16_by_9(self):
        self.assertEqual(plugin._default_params["quality"], "medium")
        self.assertEqual(plugin._default_params["image_size"], "4K")
        self.assertEqual(plugin._default_params["aspect_ratio"], "16:9")
        self.assertEqual(plugin.build_gpt_image_size("16:9", "4K"), "3840x2160")

    def test_model_family_uses_its_own_credential(self):
        params = {
            "gpt_api_key": "gpt-key",
            "gemini_api_key": "gemini-key",
            "api_key": "legacy-key",
        }
        self.assertEqual(plugin._api_key_for_model(params, "gpt-image-2"), "gpt-key")
        self.assertEqual(plugin._api_key_for_model(params, "gemini-3.1-flash-image-preview"), "gemini-key")

    def test_task_log_popup_groups_events_and_can_clear(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            plugin, "_ASYNC_TASK_LOG_PATH", Path(temp_dir) / "async_tasks.jsonl"
        ):
            plugin._record_async_task("accepted", task_id="task-1", status="queued")
            plugin._record_async_task("status", task_id="task-1", status="processing")
            plugin._record_async_task("completed", task_id="task-1", status="completed", output_count=1)

            page = plugin.handle_action("get_task_logs", {"page": 1, "page_size": 10})
            self.assertTrue(page["ok"])
            self.assertEqual(page["total"], 1)
            self.assertEqual(page["tasks"][0]["event_count"], 3)
            self.assertEqual(plugin.handle_action("open_task_logs")["open_page"], "task_log.html")

            cleared = plugin.handle_action("clear_task_logs", {"mode": "all"})
            self.assertTrue(cleared["ok"])
            self.assertEqual(plugin.handle_action("get_task_logs")["total"], 0)

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
                lambda *_: None,
            )

        self.assertEqual(accepted["task_id"], "task-contract-1")
        self.assertEqual(outputs, [{"type": "url", "value": "https://api.yaliai.com/v1/generated-images/test.png"}])
        headers = session.post_calls[0][1]["headers"]
        self.assertEqual(headers["Idempotency-Key"], request_id)
        self.assertEqual(headers["X-Request-ID"], request_id)
        self.assertEqual(len(session.get_calls), 2)

    def test_batch_uses_host_positions_and_serial_order(self):
        request_ids = []
        session = _Session()

        def fake_gemini(**kwargs):
            request_ids.append(kwargs["request_id"])
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
                "output_position": [4, 1, 7],
                "batch_num": 3,
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
            ["0002_shot-a_3_4.png", "0002_shot-a_3_1.png", "0002_shot-a_3_7.png"],
        )
        self.assertEqual(len(request_ids), 3)
        self.assertEqual(len(set(request_ids)), 3)
        self.assertTrue(all(f"_{position}_" in request_id for request_id, position in zip(request_ids, [4, 1, 7])))
        self.assertTrue(getattr(session, "closed", False))

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
