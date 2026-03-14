from __future__ import annotations

import time
import urllib.error
import urllib.request

from .cluster import CommandError
from .ui import detail, step, success


def wait_for_endpoint(
    *,
    target_url: str,
    endpoint_path: str = "/v1/models",
    timeout_seconds: int = 3600,
    retry_interval_seconds: int = 10,
    verify_tls: bool = False,
) -> None:
    target = f"{target_url.rstrip('/')}{endpoint_path}"
    deadline = time.time() + timeout_seconds
    attempt = 0
    last_status = ""

    step(f"Waiting for endpoint {target}")
    detail(
        f"Timeout: {timeout_seconds}s, retry interval: {retry_interval_seconds}s, "
        f"TLS verification: {'enabled' if verify_tls else 'disabled'}"
    )

    while time.time() < deadline:
        attempt += 1
        try:
            request = urllib.request.Request(
                target, headers={"Accept": "application/json"}
            )
            context = None
            if not verify_tls and target.startswith("https://"):
                import ssl

                context = ssl._create_unverified_context()
            with urllib.request.urlopen(
                request, timeout=30, context=context
            ) as response:
                if 200 <= response.status < 400:
                    success(
                        f"Endpoint ready after {attempt} attempt"
                        f"{'' if attempt == 1 else 's'}: {target}"
                    )
                    return
                status = f"HTTP {response.status}"
                if status != last_status:
                    detail(f"Endpoint not ready yet: {status}")
                    last_status = status
        except urllib.error.HTTPError as exc:
            status = f"HTTP {exc.code}"
            if status != last_status:
                detail(f"Endpoint not ready yet: {status}")
                last_status = status
            if 500 <= exc.code < 600:
                pass
        except Exception as exc:  # noqa: BLE001
            status = exc.__class__.__name__
            if status != last_status:
                detail(f"Endpoint not ready yet: {status}")
                last_status = status
        time.sleep(retry_interval_seconds)

    raise CommandError(f"timed out waiting for endpoint: {target}")
