"""The dashboard page is served, and the parts of it a test can honestly check.

What is verified here: the page is reachable, the assets it depends on are
reachable, the risk disclosure is present **above the data** rather than in a
footer, and the page pulls in no third-party script.

What is **not** verified here, and is not claimed: visual layout, colour
contrast, and whether the charts look right. Those need a browser. The browser
harness was not connected while this was written, so they are untested rather
than passed. The JavaScript's own behaviour — the parts that would otherwise be
guesses about rendering — is checked by ``tests/js/dashboard_render_check.mjs``,
which runs under node and covers rendering, undefined handling, and the chart
degenerate cases.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from quantforge.api.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(
        token_secret=b"test-secret-key-at-least-32-bytes-long"))


def _html(client, path="/static/dashboard.html"):
    resp = client.get(path)
    assert resp.status_code == 200, f"{path} -> {resp.status_code}"
    assert "text/html" in resp.headers.get("content-type", "")
    return resp.text


class TestServed:
    def test_dashboard_page_is_served(self, client):
        assert len(_html(client)) > 5000

    def test_index_page_is_served_at_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "QuantForge" in resp.text

    def test_manifest_and_service_worker_are_served(self, client):
        """The page registers a service worker, so a 404 there would fail
        silently in the browser and break offline use with no visible error."""
        for path in ("/static/manifest.json", "/static/sw.js"):
            assert client.get(path).status_code == 200, path

    def test_icons_are_served(self, client):
        for path in ("/static/icon-192.png", "/static/icon-512.png"):
            resp = client.get(path)
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"

    def test_page_needs_no_authentication(self, client):
        """The login shell and the risk disclosure have to be readable before
        anyone holds a token. The page contains no portfolio data; every number
        comes from a token-authenticated call."""
        assert client.get("/static/dashboard.html").status_code == 200


class TestDisclosurePlacement:
    def test_disclosure_appears_before_the_data_region(self, client):
        """A reader who screenshots one card should still carry the warning.
        A disclosure in a footer is a disclosure nobody scrolls to."""
        html = _html(client)
        disclosure = html.index('class="disclosure"')
        grid = html.index('id="grid"')
        assert disclosure < grid, (
            "the risk disclosure must appear above the rendered components, "
            "not below them"
        )

    def test_disclosure_says_results_do_not_predict_returns(self, client):
        html = _html(client)
        disclosure = html[html.index('class="disclosure"'):]
        disclosure = disclosure[:disclosure.index("</div>")]
        assert "does not predict" in disclosure
        assert "nothing here is a" in disclosure and "forecast" in disclosure

    def test_disclosure_explains_why_every_number_shows_its_sample(self, client):
        html = _html(client)
        assert "sample size" in html

    def test_main_page_still_carries_its_disclaimer(self, client):
        assert "guarantees profits" in _html(client, "/static/index.html")


class TestNoThirdPartyDependencies:
    def test_no_external_script_or_style(self, client):
        """A page pulling assets from a third-party CDN is a supply-chain
        dependency and an availability problem, for no benefit here."""
        html = _html(client)
        for pattern in (r'src\s*=\s*["\']https?://', r'href\s*=\s*["\']https?://'):
            found = re.findall(pattern, html)
            assert not found, f"external asset reference(s): {found}"

    def test_charts_are_hand_rolled_svg(self, client):
        """No chart library. The page draws its own SVG so it works offline
        once cached and carries no third-party code."""
        html = _html(client)
        assert "<svg" in html or "createElementNS" in html or "polyline" in html
        assert "chart.js" not in html.lower()
        assert "d3" not in html.lower()

    def test_no_inline_event_handlers(self, client):
        """Inline onclick attributes would need an unsafe-inline CSP carve-out
        to be blocked, which weakens the policy for the whole page."""
        html = _html(client)
        assert not re.search(r'on(click|load|error)\s*=\s*["\']', html)


class TestPageBehaviourMarkers:
    def test_unavailable_components_are_rendered_distinctly(self, client):
        """The page must be able to show a component as unavailable; that is the
        whole design, and a page that only draws successes cannot express it."""
        html = _html(client)
        assert "unavailable" in html
        assert "available: false" not in html  # it reads the JSON field
        assert "c.available" in html

    def test_undefined_values_render_as_words_not_zero(self, client):
        """A blank cell reads as 'no data' and a 0 reads as a measured zero.
        Neither means 'not computable from this sample'."""
        html = _html(client)
        assert "not defined" in html
        assert "not finite" in html

    def test_basis_is_rendered_for_each_component(self, client):
        html = _html(client)
        assert "n_observations" in html
        assert "caveat" in html

    def test_every_component_name_appears_so_none_is_orphaned(self, client):
        html = _html(client)
        for name in ("equity_curve", "daily_pnl", "unrealized_pnl",
                     "realized_pnl", "exposure", "positions", "sector_exposure",
                     "asset_class_exposure", "correlations",
                     "portfolio_volatility", "var", "drawdown",
                     "summary_statistics", "beta"):
            assert f"{name}:" in html, f"{name} has no renderer"

    def test_unclassified_exposure_has_its_own_style(self, client):
        """Exposure that could not be grouped must be visually distinct from a
        real group, or it reads as a sector."""
        html = _html(client)
        assert "unclassified" in html.lower()
