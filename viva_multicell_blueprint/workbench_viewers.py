"""Workspace-contributed Explore-tab tools (vivarium-workbench ``workbench_viewers``).

Exposes **Blueprint Studio** — a launcher tool that opens a small UI to fill in a
blueprint, pick the mechanics engine (CPM or vertex), and run it. The UI is the
self-contained page served at ``/assets/blueprint_studio.html`` (committed under
``workspace/reports/assets/``); it talks back to the workbench over the same-origin
``/api/composite-test-run`` endpoint, so no new server routes are needed.

See :mod:`vivarium_workbench.lib.analysis_viewers` for the descriptor contract.
"""
from __future__ import annotations

from pathlib import Path

_APP_REL = "assets/blueprint_studio.html"


def _studio_url(ws_root, study, run, ctx) -> dict:
    """Launch: open the Blueprint Studio page. ``ctx['public_base']`` is the
    externally-reachable base URL (may carry a reverse-proxy path prefix); fall
    back to a root-relative URL when it is absent."""
    base = (ctx or {}).get("public_base") or ""
    base = base.rstrip("/")
    return {"url": f"{base}/{_APP_REL}" if base else f"/{_APP_REL}"}


def get_viewers(ws_root) -> list[dict]:
    app = Path(ws_root) / "workspace" / "reports" / _APP_REL
    if not app.is_file():
        return []
    return [
        {
            "id": "blueprint-studio",
            "title": "Blueprint Studio",
            "description": ("Fill in a backend-neutral blueprint, pick the mechanics "
                            "engine (CPM or vertex), and run it — results land in the "
                            "Simulations DB."),
            "kind": "launcher",
            "launch": _studio_url,
            # A root-absolute target href makes the Explore-tab card clickable BOTH
            # live and in the published read-only snapshot: the frontend prefixes it
            # with the base path, so it opens the local page (Run works) on a live
            # server and the published page (view-only) under /<repo>/workbench/.
            # The target is tied to a study because the workbench only keeps a
            # target's href when it carries a study/run (env_worker._av_resolve_targets);
            # the Studio is a workspace-level tool, anchored to the capstone study.
            "targets": [{"study": "flu-lung-crossparadigm",
                         "label": "Open Blueprint Studio",
                         "href": f"/{_APP_REL}"}],
        }
    ]
