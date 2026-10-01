"""ArxivClient.get_papers / html / pdf_bytes against fake servers (no network), and the
Hugging Face project page field."""

from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from app.sources.arxiv import MAX_PDF_BYTES, ArxivClient
from app.sources.hf import HFPaper
from tests.helpers import CountingLimiter
from tests.unit.test_arxiv import FakeResponse, FakeSession, _feed

LATEXML = '<html><body><article class="ltx_document"><h1>Invented</h1></article></body></html>'


def _web_client(handler: httpx.MockTransport) -> tuple[ArxivClient, CountingLimiter]:
    limiter = CountingLimiter()
    http = httpx.Client(base_url="https://arxiv.org", transport=handler, follow_redirects=False)
    return ArxivClient(limiter=limiter, http=http, sleep=lambda s: None), limiter


def _server(routes: dict[str, httpx.Response]) -> tuple[httpx.MockTransport, list[str]]:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return routes.get(request.url.path, httpx.Response(404))

    return httpx.MockTransport(handle), seen


# --- get_papers -------------------------------------------------------------------------------


def test_get_papers_sends_one_request_per_50_ids() -> None:
    session = FakeSession(
        [FakeResponse(200, _feed(1, 50, 50)), FakeResponse(200, _feed(51, 10, 10))]
    )
    limiter = CountingLimiter()
    client = ArxivClient(limiter=limiter, session=session, sleep=lambda s: None)
    ids = [f"2601.{n:05d}" for n in range(1, 61)]
    papers = client.get_papers(ids)
    assert [p.arxiv_id for p in papers] == ids
    assert len(session.urls) == 2 and limiter.slots == 2
    first = parse_qs(urlparse(session.urls[0]).query)["id_list"][0].split(",")
    assert len(first) == 50


def test_get_papers_with_no_ids_makes_no_request() -> None:
    session = FakeSession([])
    assert ArxivClient(limiter=CountingLimiter(), session=session).get_papers([]) == []
    assert session.urls == []


# --- html -------------------------------------------------------------------------------------


def test_html_returns_the_latexml_page() -> None:
    transport, seen = _server({"/html/2601.00001v2": httpx.Response(200, text=LATEXML)})
    client, limiter = _web_client(transport)
    assert client.html("2601.00001", 2) == LATEXML
    assert seen == ["https://arxiv.org/html/2601.00001v2"] and limiter.slots == 1


def test_html_missing_or_not_latexml_is_none() -> None:
    transport, _ = _server({"/html/2601.00002v1": httpx.Response(200, text="<p>No HTML.</p>")})
    client, _ = _web_client(transport)
    assert client.html("2601.00001", 1) is None  # 404
    assert client.html("2601.00002", 1) is None  # a page, but no LaTeXML document


def test_each_redirect_hop_takes_its_own_limiter_slot() -> None:
    transport, seen = _server(
        {
            "/html/2601.00001v1": httpx.Response(301, headers={"location": "/html/2601.00001v1/"}),
            "/html/2601.00001v1/": httpx.Response(200, text=LATEXML),
        }
    )
    client, limiter = _web_client(transport)
    assert client.html("2601.00001", 1) == LATEXML
    assert len(seen) == 2 and limiter.slots == 2


def test_redirect_off_arxiv_is_refused() -> None:
    transport, seen = _server(
        {
            "/pdf/2601.00001v1": httpx.Response(
                302, headers={"location": "https://evil.example/x.pdf"}
            )
        }
    )
    client, _ = _web_client(transport)
    with pytest.raises(ValueError, match="off arxiv.org"):
        client.pdf_bytes("2601.00001", 1)
    assert len(seen) == 1


def test_transient_errors_are_retried_each_in_a_slot() -> None:
    calls = iter([httpx.Response(503), httpx.Response(200, text=LATEXML)])
    transport = httpx.MockTransport(lambda request: next(calls))
    client, limiter = _web_client(transport)
    assert client.html("2601.00001", 1) == LATEXML
    assert limiter.slots == 2


def test_invalid_ids_never_reach_the_url() -> None:
    client, _ = _web_client(httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(ValueError):
        client.html("../../etc/passwd", 1)


# --- pdf --------------------------------------------------------------------------------------


def test_pdf_bytes() -> None:
    pdf = b"%PDF-1.7 invented"
    transport, seen = _server(
        {
            "/pdf/2601.00001v3": httpx.Response(200, content=pdf),
            "/pdf/2601.00002v1": httpx.Response(200, content=b"<html>not a pdf</html>"),
            "/pdf/2601.00003v1": httpx.Response(200, content=b"%PDF" + b"0" * MAX_PDF_BYTES),
        }
    )
    client, _ = _web_client(transport)
    assert client.pdf_bytes("2601.00001", 3) == pdf
    assert client.pdf_bytes("2601.00002", 1) is None  # not a PDF
    assert client.pdf_bytes("2601.00003", 1) is None  # too large
    assert client.pdf_bytes("2601.00009", 1) is None  # 404
    assert seen[0] == "https://arxiv.org/pdf/2601.00001v3"


# --- Hugging Face project page ----------------------------------------------------------------


def test_hf_paper_reads_the_project_page() -> None:
    paper = HFPaper.model_validate({"id": "2601.00001", "projectPage": "https://example.org/p"})
    assert paper.project_page == "https://example.org/p"
    assert HFPaper.model_validate({"id": "2601.00001"}).project_page is None
