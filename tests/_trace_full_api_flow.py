"""One-off: full HVAC Optimizer HTTP API trace (run: PYTHONPATH=. python tests/_trace_full_api_flow.py)."""
from __future__ import annotations

import asyncio
import io
import json
import uuid

import httpx
import numpy as np
import pandas as pd

from hvac_optimizer.backend.main import app


def _summarize(obj: object, max_len: int = 1200) -> str:
    if isinstance(obj, dict):
        out = {k: ("<...>" if k in ("stage1", "ml") else v) for k, v in obj.items()}
        s = json.dumps(out, ensure_ascii=False, default=str)[:max_len]
    else:
        s = str(obj)[:max_len]
    return s + ("…" if len(s) >= max_len else "")


async def _run() -> None:
    site_id = f"site_trace_{uuid.uuid4().hex[:6]}"
    transport = httpx.ASGITransport(app=app)
    base = "http://testserver"
    timeout = httpx.Timeout(300.0)

    async with httpx.AsyncClient(transport=transport, base_url=base, timeout=timeout) as http:

        async def step(name: str, coro) -> object:
            print(f"\n=== {name} ===")
            r = await coro
            print(f"HTTP {r.status_code}")
            try:
                body = r.json()
            except Exception:
                body = r.text
            print(_summarize(body if isinstance(body, str) else body))
            r.raise_for_status()
            return body

        await step("0. GET /health", http.get("/health"))
        await step("1. GET /api/v1/sites/list", http.get("/api/v1/sites/list"))

        n = 240
        index = pd.date_range("2024-07-11", periods=n, freq="15min")
        hour = index.hour.to_numpy()
        outdoor_temp = 28 + 4 * np.sin(2 * np.pi * hour / 24) + np.linspace(0, 0.5, n)
        q = 420 + 10 * outdoor_temp + 0.2 * np.arange(n)
        total_power = 0.6 * q + 40
        df = pd.DataFrame(
            {
                "time": index,
                "CHWS": 7.0 + 0.15 * np.sin(2 * np.pi * np.arange(n) / 48),
                "CHWR": 12.0 + 0.2 * np.cos(2 * np.pi * np.arange(n) / 48),
                "CHW_Flow": 1000.0 + 20 * np.sin(2 * np.pi * np.arange(n) / 32),
                "CHWP_Hz": 50.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 24),
                "CT_Hz": 42.0 + 1.0 * np.cos(2 * np.pi * np.arange(n) / 20),
                "CWS": 28.0 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 36),
                "OA_Temp": outdoor_temp,
                "RH": 65.0 + 8 * np.cos(2 * np.pi * hour / 24),
                "SYS_kW": total_power,
                "CH_kW": total_power * 0.7,
                "CHWP_kW": total_power * 0.08,
                "CWP_kW": total_power * 0.07,
                "CT_kW": total_power * 0.05,
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_bytes = buf.getvalue().encode("utf-8")

        await step(
            f"2. POST /api/v1/sites/{site_id}/data/upload",
            http.post(
                f"/api/v1/sites/{site_id}/data/upload",
                files={"file": ("flow.csv", csv_bytes, "text/csv")},
            ),
        )
        await step(
            f"3. GET /api/v1/sites/{site_id}/data/diagnostics",
            http.get(f"/api/v1/sites/{site_id}/data/diagnostics"),
        )
        await step(
            f"4. GET /api/v1/sites/{site_id}/data/mapping/suggest",
            http.get(f"/api/v1/sites/{site_id}/data/mapping/suggest"),
        )
        # Use explicit mappings (suggest can mis-label columns for synthetic CSVs).
        mappings = [
            {"source": "time", "target": "timestamp"},
            {"source": "CHWS", "target": "chws_temp"},
            {"source": "CHWR", "target": "chwr_temp"},
            {"source": "CHW_Flow", "target": "chw_flow"},
            {"source": "CHWP_Hz", "target": "chwp_freq"},
            {"source": "CT_Hz", "target": "ct_fan_freq"},
            {"source": "CWS", "target": "cws_temp"},
            {"source": "OA_Temp", "target": "ambient_temp"},
            {"source": "RH", "target": "oa_rh"},
            {"source": "SYS_kW", "target": "total_power"},
            {"source": "CH_kW", "target": "chiller_power"},
            {"source": "CHWP_kW", "target": "chwp_power"},
            {"source": "CWP_kW", "target": "cwp_power"},
            {"source": "CT_kW", "target": "ct_fan_power"},
        ]

        await step(
            f"5. POST /api/v1/sites/{site_id}/data/mapping",
            http.post(
                f"/api/v1/sites/{site_id}/data/mapping",
                json={"mappings": mappings},
            ),
        )

        train_body = await step(
            f"6. POST /api/v1/sites/{site_id}/data/equipment (Stage1+ML)",
            http.post(
                f"/api/v1/sites/{site_id}/data/equipment",
                json={"chillers": [{"id": 1, "rt": 500.0, "cop": 5.5}]},
            ),
        )
        assert isinstance(train_body, dict)
        if train_body.get("status") != "success":
            raise SystemExit(f"training failed: {train_body}")
        final_site = str(train_body.get("site_id", site_id))
        print(f"\n(final site_id after rename: {final_site})")

        await step(
            "7. GET /api/v1/sites/list?completed_only=true",
            http.get("/api/v1/sites/list?completed_only=true"),
        )
        await step(
            f"8. GET /api/v1/sites/{final_site}/analysis/analytics",
            http.get(f"/api/v1/sites/{final_site}/analysis/analytics"),
        )
        opt = await step(
            f"9. POST /api/v1/sites/{final_site}/analysis/optimize",
            http.post(
                f"/api/v1/sites/{final_site}/analysis/optimize",
                json={
                    "chws": [6.0, 9.0],
                    "chwp": [40.0, 55.0],
                    "ct_fan": [35.0, 50.0],
                    "cws": [26.0, 30.0],
                },
            ),
        )
        assert isinstance(opt, dict)
        print("\n(optimize top-level keys):", list(opt.keys()))

        await step(
            f"10. GET /api/v1/sites/{final_site}/analysis/realtime",
            http.get(f"/api/v1/sites/{final_site}/analysis/realtime"),
        )

    print("\nDone.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
