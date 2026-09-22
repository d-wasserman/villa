import os
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from torch.utils.data import DataLoader

    import bench  # noqa: F401  (installs the guard)
    from bench import no_upload_guard
except ImportError:  # CPU utility environments without torch
    no_upload_guard = None


@unittest.skipIf(no_upload_guard is None, "torch and inference dependencies not installed")
class GuardTests(unittest.TestCase):
    def test_guard_active_after_import(self):
        self.assertTrue(no_upload_guard.is_active())

    def test_boto3_clients_blocked(self):
        import boto3

        with self.assertRaises(no_upload_guard.UploadBlockedError):
            boto3.client("s3")
        with self.assertRaises(no_upload_guard.UploadBlockedError):
            boto3.resource("s3")
        with self.assertRaises(no_upload_guard.UploadBlockedError):
            boto3.session.Session().client("s3")

    def test_s3fs_writes_blocked(self):
        try:
            import s3fs
        except ImportError:
            self.skipTest("s3fs not installed")
        fs = s3fs.S3FileSystem(anon=True)
        with self.assertRaises(no_upload_guard.UploadBlockedError):
            fs.open("s3://bucket/key", "wb")
        with self.assertRaises(no_upload_guard.UploadBlockedError):
            fs.put_file("/tmp/x", "s3://bucket/key")

    def test_external_connection_blocked_and_recorded(self):
        no_upload_guard.reset_attempts()
        with socket.socket() as s, self.assertRaises(no_upload_guard.NetworkBlockedError):
            s.connect(("203.0.113.7", 9))  # TEST-NET-3, never routed
        self.assertIn("203.0.113.7:9", no_upload_guard.network_report()["hosts_blocked"])

    def test_loopback_allowed_but_proxy_is_external(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            with socket.socket() as client:
                client.connect(("127.0.0.1", port))
            env = {k: "" for k in no_upload_guard.PROXY_ENV_VARS}
            env["HTTPS_PROXY"] = f"http://127.0.0.1:{port}"
            try:
                with mock.patch.dict(os.environ, env):
                    no_upload_guard.install([])
                    with socket.socket() as client, self.assertRaises(no_upload_guard.NetworkBlockedError):
                        client.connect(("127.0.0.1", port))
            finally:
                no_upload_guard.install([])

    def test_worker_init_installs_guard_in_spawned_worker(self):
        from tests.guard_probe import GuardProbeDataset

        def names(worker_init_fn):
            loader = DataLoader(GuardProbeDataset(), batch_size=None, num_workers=1,
                                multiprocessing_context="spawn", worker_init_fn=worker_init_fn)
            return list(loader)

        self.assertEqual(names(no_upload_guard.worker_init), ["_guarded_connect"])
        # Control: without worker_init the spawned worker is unguarded.
        self.assertEqual(names(None), ["connect"])


if __name__ == "__main__":
    unittest.main()
