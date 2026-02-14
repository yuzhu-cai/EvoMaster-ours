"""Kubernetes experiment execution service.

This runner keeps K8S job lifecycle management outside the Agent tool layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import re
import shlex
import time
from pathlib import Path
from typing import Any

from evomaster.agent.session import BaseSession
import yaml


class K8SExperimentRunner:
    """Run experiment jobs on Kubernetes and collect basic outputs."""

    def __init__(self, session: BaseSession, config: dict[str, Any] | None = None):
        self.session = session
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        self.namespace = self.config.get("namespace", "default")
        self.poll_interval_seconds = int(self.config.get("poll_interval_seconds", 15))
        self.job_timeout_seconds = int(self.config.get("job_timeout_seconds", 3600))
        self.cleanup_on_success = bool(self.config.get("cleanup_on_success", False))
        debug_pod_cfg = self.config.get("debug_pod", {})
        self.debug_pod_config = debug_pod_cfg if isinstance(debug_pod_cfg, dict) else {}

    def run(
        self,
        manifest_path: str,
        job_name: str,
        manifest_env: dict[str, str] | None = None,
        workspace_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit and wait for one K8S job."""
        prepared_manifest = self.prepare_manifest(
            manifest_path=manifest_path,
            job_name=job_name,
            manifest_env=manifest_env,
            workspace_context=workspace_context,
        )
        submit_result = self.submit_job(prepared_manifest)
        wait_result = self.wait_for_job(job_name)
        logs = self.fetch_job_logs(job_name)

        result = {
            "manifest_path": manifest_path,
            "prepared_manifest_path": prepared_manifest,
            "job_name": job_name,
            "namespace": self.namespace,
            "submit": submit_result,
            "wait": wait_result,
            "logs": logs,
        }

        if wait_result.get("status") == "succeeded" and self.cleanup_on_success:
            result["cleanup"] = self.cleanup_job(job_name)

        return result

    def is_debug_pod_enabled(self) -> bool:
        """Whether debug pod execution is enabled for debug_test."""
        return bool(self.debug_pod_config.get("enabled", False))

    def exec_debug_command(
        self,
        command: str,
        timeout: int,
        working_dir: str,
        env_init: str = "",
        workspace_path: str | None = None,
    ) -> dict[str, Any]:
        """Execute command in debug pod, creating/reusing pod when needed."""
        if not self.is_debug_pod_enabled():
            raise RuntimeError("k8s_runner.debug_pod.enabled is false")

        host_workspace = self._resolve_debug_workspace_host_path(workspace_path)
        ensure_result = self.ensure_debug_pod(host_workspace)
        pod_name = str(ensure_result.get("pod_name", "")).strip()
        if not pod_name:
            raise RuntimeError("failed to resolve debug pod name")

        container_workspace = str(
            self.debug_pod_config.get("workspace_mount_path", "/workspace")
        ).strip() or "/workspace"
        container_working_dir = self._resolve_container_working_dir(
            host_working_dir=working_dir,
            host_workspace=str(host_workspace),
            container_workspace=container_workspace,
        )

        cmd_parts: list[str] = []
        if env_init.strip():
            cmd_parts.append(env_init.strip())
        cmd_parts.append(f"cd {shlex.quote(container_working_dir)}")
        cmd_parts.append(command)
        exec_script = " && ".join(cmd_parts)

        exec_cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} exec {shlex.quote(pod_name)}"
        )
        container_name = str(self.debug_pod_config.get("container_name", "")).strip()
        if container_name:
            exec_cmd += f" -c {shlex.quote(container_name)}"
        exec_cmd += f" -- /bin/sh -lc {shlex.quote(exec_script)}"

        result = self._run_command(exec_cmd, timeout=max(int(timeout), 5))
        result.update(
            {
                "mode": "k8s_debug_pod",
                "namespace": self.namespace,
                "pod_name": pod_name,
                "container_working_dir": container_working_dir,
                "host_workspace": str(host_workspace),
            }
        )

        if bool(self.debug_pod_config.get("cleanup_after_exec", False)):
            result["cleanup"] = self.cleanup_debug_pod(pod_name, wait=False)
        return result

    def ensure_debug_pod(self, workspace_path: str | Path | None = None) -> dict[str, Any]:
        """Ensure debug pod exists and is Ready."""
        if not self.is_debug_pod_enabled():
            raise RuntimeError("k8s_runner.debug_pod.enabled is false")

        host_workspace = self._resolve_debug_workspace_host_path(workspace_path)
        pod_name = self._build_debug_pod_name(host_workspace)

        status = self.get_pod_status(pod_name)
        phase = status.get("status", "unknown")
        if phase == "running":
            return {
                "status": "ready",
                "pod_name": pod_name,
                "namespace": self.namespace,
                "source": "existing",
            }
        if phase == "pending":
            ready_timeout = int(self.debug_pod_config.get("ready_timeout_seconds", 120))
            wait_result = self.wait_for_pod_ready(pod_name, ready_timeout_seconds=ready_timeout)
            if wait_result.get("status") == "ready":
                return {
                    "status": "ready",
                    "pod_name": pod_name,
                    "namespace": self.namespace,
                    "source": "existing",
                }

        if phase in {"failed", "succeeded"}:
            self.cleanup_debug_pod(pod_name, wait=False)

        manifest_path = self.prepare_debug_pod_manifest(
            pod_name=pod_name,
            workspace_host_path=host_workspace,
        )
        apply_cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} apply -f {shlex.quote(manifest_path)}"
        )
        apply_result = self._run_command(apply_cmd, timeout=120)
        if apply_result.get("exit_code", 1) != 0:
            raise RuntimeError(
                "failed to apply debug pod manifest: "
                + (apply_result.get("stderr", "") or apply_result.get("output", ""))
            )

        ready_timeout = int(self.debug_pod_config.get("ready_timeout_seconds", 120))
        wait_result = self.wait_for_pod_ready(pod_name, ready_timeout_seconds=ready_timeout)
        if wait_result.get("status") != "ready":
            detail = (
                wait_result.get("wait_stderr", "")
                or wait_result.get("stderr", "")
                or wait_result.get("output", "")
            )
            describe_output = str(wait_result.get("describe", "")).strip()
            if describe_output:
                detail = f"{detail}\n\n[describe]\n{describe_output}".strip()
            raise RuntimeError(
                "debug pod is not ready: " + detail
            )

        return {
            "status": "ready",
            "pod_name": pod_name,
            "namespace": self.namespace,
            "source": "created",
            "manifest_path": manifest_path,
        }

    def prepare_debug_pod_manifest(
        self,
        pod_name: str,
        workspace_host_path: str | Path,
    ) -> str:
        """Render debug pod manifest for a workspace host path."""
        cfg = self.debug_pod_config
        image = str(cfg.get("image", "")).strip()
        if not image:
            raise ValueError("k8s_runner.debug_pod.image is required")

        workspace_host = Path(workspace_host_path).expanduser().resolve()
        workspace_host.mkdir(parents=True, exist_ok=True)
        workspace_mount_path = str(cfg.get("workspace_mount_path", "/workspace")).strip() or "/workspace"

        workspace = Path(getattr(self.session.config, "workspace_path", ".")).resolve()
        out_dir = workspace / ".embomaster" / "k8s_manifests"
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{pod_name}.debug.yaml"

        command = self._normalize_debug_pod_command(cfg.get("command"))
        raw_args = cfg.get("args")
        args: list[str] | None = None
        if isinstance(raw_args, list):
            args = [str(x) for x in raw_args if str(x).strip()]

        env_map = cfg.get("env", {})
        env_vars: list[dict[str, str]] = []
        if isinstance(env_map, dict):
            env_vars = [{"name": str(k), "value": str(v)} for k, v in sorted(env_map.items())]

        container_name = str(cfg.get("container_name", "debug")).strip() or "debug"
        labels = {"app": "embomaster-debug", "managed-by": "embomaster"}
        raw_labels = cfg.get("labels", {})
        if isinstance(raw_labels, dict):
            for key, value in raw_labels.items():
                labels[str(key)] = str(value)

        container: dict[str, Any] = {
            "name": container_name,
            "image": image,
            "imagePullPolicy": str(cfg.get("image_pull_policy", "IfNotPresent")),
            "workingDir": workspace_mount_path,
            "command": command,
            "volumeMounts": [{"name": "workspace", "mountPath": workspace_mount_path}],
        }
        if args:
            container["args"] = args
        if env_vars:
            container["env"] = env_vars

        spec: dict[str, Any] = {
            "restartPolicy": "Always",
            "containers": [container],
            "volumes": [
                {
                    "name": "workspace",
                    "hostPath": {
                        "path": str(workspace_host),
                        "type": str(cfg.get("workspace_host_path_type", "Directory")),
                    },
                }
            ],
        }
        node_name = self._resolve_debug_node_name()
        if node_name:
            spec["nodeName"] = node_name
        node_selector = cfg.get("node_selector")
        if isinstance(node_selector, dict) and node_selector:
            spec["nodeSelector"] = {str(k): str(v) for k, v in node_selector.items()}
        tolerations = cfg.get("tolerations")
        if isinstance(tolerations, list) and tolerations:
            spec["tolerations"] = tolerations

        pod_doc: dict[str, Any] = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": self.namespace,
                "labels": labels,
            },
            "spec": spec,
        }

        with dst.open("w", encoding="utf-8") as f:
            yaml.safe_dump(pod_doc, f, sort_keys=False, allow_unicode=True)
        return str(dst)

    def get_pod_status(self, pod_name: str) -> dict[str, Any]:
        """Get pod phase from kubectl."""
        cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} get pod "
            f"{shlex.quote(pod_name)} -o json"
        )
        result = self._run_command(cmd, timeout=60)
        if result.get("exit_code", 1) != 0:
            stderr = result.get("stderr", "") or result.get("output", "")
            if "NotFound" in stderr:
                return {
                    "status": "not_found",
                    "pod_name": pod_name,
                    "namespace": self.namespace,
                    "error": stderr,
                }
            return {
                "status": "unknown",
                "pod_name": pod_name,
                "namespace": self.namespace,
                "error": stderr,
            }

        raw = result.get("stdout", "").strip()
        try:
            payload = json.loads(raw)
        except Exception:
            return {
                "status": "unknown",
                "pod_name": pod_name,
                "namespace": self.namespace,
                "error": "invalid kubectl json output",
                "raw": raw[:1000],
            }

        phase_raw = str(payload.get("status", {}).get("phase", "")).strip().lower()
        phase_map = {
            "running": "running",
            "pending": "pending",
            "succeeded": "succeeded",
            "failed": "failed",
        }
        return {
            "status": phase_map.get(phase_raw, "unknown"),
            "pod_name": pod_name,
            "namespace": self.namespace,
            "details": payload.get("status", {}),
        }

    def wait_for_pod_ready(self, pod_name: str, ready_timeout_seconds: int = 120) -> dict[str, Any]:
        """Wait until pod becomes Ready."""
        cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} wait "
            f"--for=condition=Ready pod/{shlex.quote(pod_name)} "
            f"--timeout={int(ready_timeout_seconds)}s"
        )
        result = self._run_command(cmd, timeout=max(int(ready_timeout_seconds) + 10, 30))
        if result.get("exit_code", 1) == 0:
            return {
                "status": "ready",
                "pod_name": pod_name,
                "namespace": self.namespace,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "output": result.get("output", ""),
            }
        status = self.get_pod_status(pod_name)
        status["describe"] = self.describe_pod(pod_name).get("output", "")
        status.update(
            {
                "wait_stdout": result.get("stdout", ""),
                "wait_stderr": result.get("stderr", ""),
                "output": result.get("output", ""),
            }
        )
        return status

    def cleanup_debug_pod(self, pod_name: str, wait: bool = False) -> dict[str, Any]:
        """Delete debug pod."""
        cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} delete pod "
            f"{shlex.quote(pod_name)} --ignore-not-found"
        )
        if not wait:
            cmd += " --wait=false"
        return self._run_command(cmd, timeout=60)

    def describe_pod(self, pod_name: str) -> dict[str, Any]:
        cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} describe pod "
            f"{shlex.quote(pod_name)}"
        )
        return self._run_command(cmd, timeout=60)

    def prepare_manifest(
        self,
        manifest_path: str,
        job_name: str,
        manifest_env: dict[str, str] | None = None,
        workspace_context: dict[str, Any] | None = None,
    ) -> str:
        """Render a run-specific job manifest with unique job name and env overrides."""
        src = Path(manifest_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"K8S manifest not found: {src}")

        workspace = Path(getattr(self.session.config, "workspace_path", ".")).resolve()
        out_dir = workspace / ".embomaster" / "k8s_manifests"
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{job_name}.yaml"

        with src.open("r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))

        patched_docs: list[dict[str, Any] | None] = []
        for doc in docs:
            if not isinstance(doc, dict):
                patched_docs.append(doc)
                continue

            kind = str(doc.get("kind", "")).lower()
            if kind == "job":
                metadata = doc.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["name"] = job_name
                    metadata["namespace"] = self.namespace

                if manifest_env:
                    self._merge_env_to_containers(doc, manifest_env)
                if workspace_context:
                    self._patch_workspace_mounts(doc, workspace_context)

            patched_docs.append(doc)

        with dst.open("w", encoding="utf-8") as f:
            yaml.safe_dump_all(patched_docs, f, sort_keys=False, allow_unicode=True)

        return str(dst)

    def _merge_env_to_containers(self, job_doc: dict[str, Any], env_map: dict[str, str]) -> None:
        """Upsert env vars to all job containers."""
        try:
            spec = job_doc.setdefault("spec", {})
            template = spec.setdefault("template", {})
            pod_spec = template.setdefault("spec", {})
            containers = pod_spec.get("containers", [])
            if not isinstance(containers, list):
                return

            for container in containers:
                if not isinstance(container, dict):
                    continue
                existing_env = container.get("env", [])
                if not isinstance(existing_env, list):
                    existing_env = []

                kv = {}
                for item in existing_env:
                    if isinstance(item, dict) and "name" in item:
                        kv[str(item["name"])] = str(item.get("value", ""))
                for key, value in env_map.items():
                    kv[str(key)] = str(value)
                container["env"] = [{"name": k, "value": v} for k, v in sorted(kv.items())]
        except Exception as e:
            self.logger.warning("Failed to merge env into manifest: %s", e)

    def _patch_workspace_mounts(
        self, job_doc: dict[str, Any], workspace_context: dict[str, Any]
    ) -> None:
        """Patch manifest to mount isolated round-workspace codebase/submission/large dirs."""
        codebase_path_raw = str(
            workspace_context.get("workspace_codebase_path", "")
        ).strip()
        if not codebase_path_raw:
            return

        codebase_host_path = Path(codebase_path_raw).expanduser().resolve()
        if not codebase_host_path.exists():
            self.logger.warning("workspace_codebase_path does not exist: %s", codebase_host_path)
            return

        pod_spec = self._get_pod_spec(job_doc)
        if pod_spec is None:
            self.logger.warning("Failed to locate pod spec in job manifest")
            return

        containers = self._get_target_containers(pod_spec)
        if not containers:
            self.logger.warning("No target containers found in job manifest")
            return

        volumes = pod_spec.setdefault("volumes", [])
        if not isinstance(volumes, list):
            volumes = []
            pod_spec["volumes"] = volumes

        container_workspace = str(self.config.get("container_workspace", "/workspace")).strip()
        codebase_mount_path = str(
            self.config.get("codebase_mount_path", f"{container_workspace}/codebase")
        ).strip()
        codebase_volume_name = str(
            self.config.get("codebase_volume_name", "codebase-volume")
        ).strip()

        self._upsert_volume(
            volumes,
            {
                "name": codebase_volume_name,
                "hostPath": {"path": str(codebase_host_path), "type": "Directory"},
            },
        )
        for container in containers:
            volume_mounts = self._ensure_volume_mounts(container)
            self._upsert_volume_mount(
                volume_mounts,
                {"name": codebase_volume_name, "mountPath": codebase_mount_path},
            )

        enable_submission_mount = bool(self.config.get("enable_submission_mount", True))
        if enable_submission_mount:
            submission_path_raw = str(workspace_context.get("submission_dir", "")).strip()
            if submission_path_raw:
                submission_host_path = Path(submission_path_raw).expanduser().resolve()
            else:
                submission_host_path = codebase_host_path / "submission"
            submission_host_path.mkdir(parents=True, exist_ok=True)

            submission_volume_name = str(
                self.config.get("submission_volume_name", "submission")
            ).strip()
            submission_mount_path = str(
                self.config.get("submission_mount_path", f"{container_workspace}/submission")
            ).strip()

            self._upsert_volume(
                volumes,
                {
                    "name": submission_volume_name,
                    "hostPath": {"path": str(submission_host_path), "type": "Directory"},
                },
            )
            for container in containers:
                volume_mounts = self._ensure_volume_mounts(container)
                self._upsert_volume_mount(
                    volume_mounts,
                    {"name": submission_volume_name, "mountPath": submission_mount_path},
                )

        enable_large_dir_mounts = bool(self.config.get("enable_large_dir_mounts", True))
        large_dirs = workspace_context.get("workspace_large_dirs", [])
        if not enable_large_dir_mounts or not isinstance(large_dirs, list):
            return

        volume_prefix = str(self.config.get("large_dir_volume_prefix", "large-dir")).strip() or "large-dir"
        for idx, item in enumerate(large_dirs):
            if not isinstance(item, dict):
                continue
            src = str(item.get("src", "")).strip()
            rel = str(item.get("rel", "")).strip().strip("/")
            if not src or not rel:
                continue

            src_path = Path(src).expanduser().resolve()
            if not src_path.exists():
                self.logger.warning("Skip missing large dir mount source: %s", src_path)
                continue

            mount_path = f"{codebase_mount_path.rstrip('/')}/{rel}"
            volume_name = self._safe_volume_name(f"{volume_prefix}-{idx}")

            self._upsert_volume(
                volumes,
                {"name": volume_name, "hostPath": {"path": str(src_path), "type": "Directory"}},
            )
            for container in containers:
                volume_mounts = self._ensure_volume_mounts(container)
                self._upsert_volume_mount(
                    volume_mounts,
                    {"name": volume_name, "mountPath": mount_path},
                )

    def _get_pod_spec(self, job_doc: dict[str, Any]) -> dict[str, Any] | None:
        try:
            spec = job_doc.setdefault("spec", {})
            template = spec.setdefault("template", {})
            pod_spec = template.setdefault("spec", {})
            if not isinstance(pod_spec, dict):
                return None
            return pod_spec
        except Exception:
            return None

    def _get_target_containers(self, pod_spec: dict[str, Any]) -> list[dict[str, Any]]:
        containers = pod_spec.get("containers", [])
        if not isinstance(containers, list):
            return []

        normalized = [c for c in containers if isinstance(c, dict)]
        if not normalized:
            return []

        target_name = str(self.config.get("container_name", "")).strip()
        if target_name:
            matched = [c for c in normalized if str(c.get("name", "")) == target_name]
            if matched:
                return matched
            self.logger.warning(
                "Configured container_name=%s not found; fallback to first container",
                target_name,
            )

        if bool(self.config.get("apply_mounts_to_all_containers", False)):
            return normalized
        return [normalized[0]]

    def _ensure_volume_mounts(self, container: dict[str, Any]) -> list[dict[str, Any]]:
        volume_mounts = container.get("volumeMounts", [])
        if not isinstance(volume_mounts, list):
            volume_mounts = []
            container["volumeMounts"] = volume_mounts
        return volume_mounts

    def _upsert_volume(self, volumes: list[dict[str, Any]], item: dict[str, Any]) -> None:
        name = str(item.get("name", "")).strip()
        if not name:
            return
        for index, volume in enumerate(volumes):
            if isinstance(volume, dict) and str(volume.get("name", "")) == name:
                volumes[index] = item
                return
        volumes.append(item)

    def _upsert_volume_mount(
        self, volume_mounts: list[dict[str, Any]], item: dict[str, Any]
    ) -> None:
        name = str(item.get("name", "")).strip()
        if not name:
            return
        for index, mount in enumerate(volume_mounts):
            if isinstance(mount, dict) and str(mount.get("name", "")) == name:
                volume_mounts[index] = item
                return
        volume_mounts.append(item)

    def _safe_volume_name(self, value: str) -> str:
        text = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
        if not text:
            return "mount"
        return text[:63]

    def submit_job(self, manifest_path: str) -> dict[str, Any]:
        cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} apply -f "
            f"{shlex.quote(manifest_path)}"
        )
        return self._run_command(cmd, timeout=120)

    def wait_for_job(self, job_name: str) -> dict[str, Any]:
        start = time.monotonic()
        while True:
            status = self.get_job_status(job_name)
            phase = status.get("status", "unknown")
            if phase in {"succeeded", "failed"}:
                return status

            if time.monotonic() - start > self.job_timeout_seconds:
                return {
                    "status": "timeout",
                    "job_name": job_name,
                    "namespace": self.namespace,
                    "reason": f"timeout after {self.job_timeout_seconds}s",
                }

            time.sleep(self.poll_interval_seconds)

    def get_job_status(self, job_name: str) -> dict[str, Any]:
        cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} get job "
            f"{shlex.quote(job_name)} -o json"
        )
        result = self._run_command(cmd, timeout=60)
        if result.get("exit_code", 1) != 0:
            return {
                "status": "unknown",
                "job_name": job_name,
                "namespace": self.namespace,
                "error": result.get("stderr", "failed to query job"),
            }

        raw = result.get("stdout", "").strip()
        try:
            payload = json.loads(raw)
        except Exception:
            return {
                "status": "unknown",
                "job_name": job_name,
                "namespace": self.namespace,
                "error": "invalid kubectl json output",
                "raw": raw[:1000],
            }

        status_obj = payload.get("status", {})
        if status_obj.get("succeeded", 0) > 0:
            phase = "succeeded"
        elif status_obj.get("failed", 0) > 0:
            phase = "failed"
        else:
            phase = "running"

        return {
            "status": phase,
            "job_name": job_name,
            "namespace": self.namespace,
            "details": status_obj,
        }

    def fetch_job_logs(self, job_name: str, tail: int = 500) -> dict[str, Any]:
        cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} logs job/{shlex.quote(job_name)} "
            f"--tail={int(tail)}"
        )
        return self._run_command(cmd, timeout=120)

    def cleanup_job(self, job_name: str) -> dict[str, Any]:
        cmd = (
            f"kubectl -n {shlex.quote(self.namespace)} delete job "
            f"{shlex.quote(job_name)} --ignore-not-found"
        )
        return self._run_command(cmd, timeout=60)

    def _resolve_debug_workspace_host_path(self, workspace_path: str | Path | None) -> Path:
        if workspace_path:
            host_workspace = Path(workspace_path).expanduser().resolve()
        else:
            host_workspace = Path(getattr(self.session.config, "workspace_path", ".")).resolve()
        host_workspace.mkdir(parents=True, exist_ok=True)
        return host_workspace

    def _resolve_debug_node_name(self) -> str:
        configured = str(self.debug_pod_config.get("node_name", "")).strip()
        if configured:
            return configured
        if not bool(self.debug_pod_config.get("pin_to_local_node", True)):
            return ""
        host = socket.gethostname().strip()
        if not host:
            return ""
        return host.split(".")[0]

    def _build_debug_pod_name(self, workspace_host_path: Path) -> str:
        pod_name = str(self.debug_pod_config.get("pod_name", "")).strip()
        if pod_name:
            return self._normalize_k8s_name(pod_name)

        prefix = str(self.debug_pod_config.get("pod_name_prefix", "embomaster-debug")).strip()
        if not prefix:
            prefix = "embomaster-debug"
        digest = hashlib.sha1(str(workspace_host_path).encode("utf-8")).hexdigest()[:8]
        return self._normalize_k8s_name(f"{prefix}-{digest}")

    def _normalize_k8s_name(self, value: str) -> str:
        text = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
        if not text:
            return "embomaster-debug"
        return text[:63]

    def _normalize_debug_pod_command(self, raw: Any) -> list[str]:
        if isinstance(raw, list):
            normalized = [str(x) for x in raw if str(x).strip()]
            if normalized:
                return normalized
        if isinstance(raw, str) and raw.strip():
            return ["/bin/sh", "-lc", raw.strip()]
        return ["sleep", "infinity"]

    def _resolve_container_working_dir(
        self,
        host_working_dir: str,
        host_workspace: str,
        container_workspace: str,
    ) -> str:
        host_wd = Path(host_working_dir).expanduser().resolve()
        host_ws = Path(host_workspace).expanduser().resolve()
        try:
            rel = host_wd.relative_to(host_ws).as_posix().strip("/")
        except ValueError:
            fallback = str(
                self.debug_pod_config.get("default_working_dir", container_workspace)
            ).strip()
            return fallback or container_workspace

        if not rel:
            return container_workspace
        return f"{container_workspace.rstrip('/')}/{rel}"

    def _run_command(self, command: str, timeout: int) -> dict[str, Any]:
        self.logger.debug("K8S command: %s", command)
        result = self.session.exec_bash(command=command, timeout=timeout)
        return {
            "command": command,
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "output": result.get("output", ""),
            "exit_code": result.get("exit_code", -1),
        }
