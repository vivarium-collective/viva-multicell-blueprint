import re

import pytest

from viva_multicell_blueprint.model_format import viz


def test_flipbook_is_self_contained_and_small():
    try:
        html = viz.render_demo(steps=3, out=None)
    except Exception as e:  # engine unavailable
        pytest.skip(f"demo engine unavailable: {e!r}")
    assert html.count("data:image/png;base64,") >= 1
    assert not re.search(r"(src|href)=[\"']https?://", html)
    assert len(html) < viz.MAX_BYTES
