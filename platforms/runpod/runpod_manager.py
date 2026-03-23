"""RunPod GPU pod lifecycle management for autoresearch.

Wraps the RunPod Python SDK to provide simple create/stop/terminate
operations with sensible defaults for autoresearch workloads.

Usage:
    from platforms.runpod.runpod_manager import RunPodManager

    mgr = RunPodManager()  # reads RUNPOD_API_KEY env var
    pod = mgr.create_pod(gpu_type="NVIDIA RTX 4090")
    print(mgr.get_jupyter_url(pod["id"]))
    # ... run experiments ...
    mgr.stop_pod(pod["id"])      # pause billing
    mgr.terminate_pod(pod["id"]) # destroy (network volume persists)
"""

import os
import time

try:
    import runpod
except ImportError:
    runpod = None

# Default Docker image for autoresearch pods
DEFAULT_IMAGE = "runpod/pytorch:2.6.0-py3.11-cuda12.8.1-devel-ubuntu22.04"
DEFAULT_VOLUME_GB = 50
DEFAULT_DISK_GB = 20


class RunPodManager:
    """Manage RunPod GPU pods for autoresearch experiments."""

    def __init__(self, api_key: str | None = None):
        if runpod is None:
            raise ImportError(
                "RunPod SDK not installed. Run: pip install runpod"
            )
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No RunPod API key. Set RUNPOD_API_KEY env var or pass api_key="
            )
        runpod.api_key = self.api_key

    def check_balance(self) -> float:
        """Return current RunPod account balance in USD."""
        myself = runpod.get_myself()
        return float(myself.get("currentSpendPerHr", 0))

    def list_gpus(self) -> list[dict]:
        """List available GPU types with pricing."""
        gpu_types = runpod.get_gpus()
        results = []
        for gpu in gpu_types:
            results.append({
                "id": gpu.get("id", ""),
                "name": gpu.get("displayName", gpu.get("id", "")),
                "vram_gb": gpu.get("memoryInGb", 0),
                "max_gpu_count": gpu.get("maxGpuCount", 1),
                "community_price": gpu.get("communityPrice", 0),
                "secure_price": gpu.get("securePrice", 0),
                "community_spot_price": gpu.get("communitySpotPrice", 0),
            })
        return sorted(results, key=lambda x: x.get("community_price", 999))

    def create_pod(
        self,
        gpu_type: str = "NVIDIA GeForce RTX 4090",
        network_volume_id: str | None = None,
        spot: bool = False,
        name: str | None = None,
        image: str = DEFAULT_IMAGE,
        volume_gb: int = DEFAULT_VOLUME_GB,
        disk_gb: int = DEFAULT_DISK_GB,
        env_vars: dict | None = None,
    ) -> dict:
        """Create a GPU pod configured for autoresearch.

        Args:
            gpu_type: GPU model name or RunPod type ID.
            network_volume_id: Persistent network volume ID (recommended).
            spot: Use spot/interruptible instance for lower cost.
            name: Pod display name.
            image: Docker image to use.
            volume_gb: Container volume size in GB (persists across stop/restart).
            disk_gb: Container disk size in GB.
            env_vars: Additional environment variables.

        Returns:
            Pod info dict with 'id', 'desiredStatus', etc.
        """
        name = name or f"autoresearch-{time.strftime('%b%d-%H%M').lower()}"

        default_env = {
            "AUTORESEARCH_BACKEND": "cuda",
            "AUTORESEARCH_CACHE_DIR": "/workspace/.cache/autoresearch",
            "PYTHONPATH": "/workspace/autoresearch-unified",
        }
        if env_vars:
            default_env.update(env_vars)

        kwargs = {
            "name": name,
            "image_name": image,
            "gpu_type_id": gpu_type,
            "cloud_type": "ALL",
            "gpu_count": 1,
            "volume_in_gb": volume_gb,
            "container_disk_in_gb": disk_gb,
            "ports": "8888/http,22/tcp",
            "env": default_env,
        }

        if network_volume_id:
            kwargs["network_volume_id"] = network_volume_id

        if spot:
            kwargs["bid_per_gpu"] = 0.0  # auto-bid at market rate

        pod = runpod.create_pod(**kwargs)
        return pod

    def get_pod(self, pod_id: str) -> dict:
        """Get current pod status."""
        return runpod.get_pod(pod_id)

    def get_jupyter_url(self, pod_id: str) -> str:
        """Return the JupyterLab URL for a running pod."""
        return f"https://{pod_id}-8888.proxy.runpod.net"

    def get_ssh_command(self, pod_id: str) -> str:
        """Return the SSH command for a running pod."""
        pod = self.get_pod(pod_id)
        runtime = pod.get("runtime", {})
        ports = runtime.get("ports", [])
        for port in ports:
            if port.get("privatePort") == 22:
                ip = port.get("ip", "")
                public_port = port.get("publicPort", 22)
                return f"ssh root@{ip} -p {public_port}"
        return f"# SSH info not yet available for pod {pod_id}"

    def wait_for_ready(self, pod_id: str, timeout: int = 300) -> dict:
        """Wait for pod to reach RUNNING status.

        Args:
            pod_id: Pod ID to wait for.
            timeout: Max seconds to wait.

        Returns:
            Pod info dict once running.

        Raises:
            TimeoutError if pod doesn't start in time.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            pod = self.get_pod(pod_id)
            status = pod.get("desiredStatus", "")
            runtime = pod.get("runtime", {})
            if status == "RUNNING" and runtime:
                return pod
            time.sleep(5)
        raise TimeoutError(f"Pod {pod_id} did not start within {timeout}s")

    def stop_pod(self, pod_id: str) -> None:
        """Stop a pod (preserves volume, stops compute billing)."""
        runpod.stop_pod(pod_id)

    def terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod (destroys container, network volume persists)."""
        runpod.terminate_pod(pod_id)

    def estimate_cost(
        self,
        hourly_rate: float,
        num_experiments: int = 80,
        minutes_per_experiment: float = 5.5,
    ) -> dict:
        """Estimate cost for a run.

        Args:
            hourly_rate: GPU hourly rate in USD.
            num_experiments: Number of experiments to run.
            minutes_per_experiment: Average time per experiment (training + overhead).

        Returns:
            Dict with hours, cost estimate.
        """
        hours = (num_experiments * minutes_per_experiment) / 60
        return {
            "num_experiments": num_experiments,
            "estimated_hours": round(hours, 1),
            "estimated_cost": round(hours * hourly_rate, 2),
        }
