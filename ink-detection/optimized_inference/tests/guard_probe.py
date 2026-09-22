"""
Dataset used by test_bench to check the guard inside spawned DataLoader
workers. Deliberately does not import bench, so the only way the guard can be
active in the worker is through bench.no_upload_guard.worker_init.
"""
import socket


class GuardProbeDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return socket.socket.connect.__name__
