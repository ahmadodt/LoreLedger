import pytest

BeautifulSoup = pytest.importorskip("bs4").BeautifulSoup

from novel_memory.scraper import extract_chapter_text, extract_next_chapter_link, fetch_page


def test_fetch_page_decodes_royalroad_html_as_utf8(monkeypatch):
    class FakeResponse:
        encoding = "ISO-8859-1"
        content = "<html><body><p>“quoted”</p></body></html>".encode("utf-8")

        def raise_for_status(self):
            return None

        @property
        def text(self):
            return self.content.decode(self.encoding)

    def fake_get(url, timeout, headers):
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)

    soup = fetch_page("https://www.royalroad.com/example")

    assert soup.get_text(strip=True) == "“quoted”"


def test_extract_chapter_text_from_royalroad_markup():
    soup = BeautifulSoup(
        """
        <html><body>
          <h1>Chapter 1: Start</h1>
          <div class="chapter-inner chapter-content">
            <p>Chapter 1: Start</p>
            <p>First paragraph.</p>
            <p>Second<br>paragraph.</p>
          </div>
        </body></html>
        """,
        "html.parser",
    )

    assert extract_chapter_text(soup) == "First paragraph.\n\nSecond paragraph."


def test_extract_next_chapter_link_makes_absolute_url():
    soup = BeautifulSoup(
        '<a href="/fiction/1/book/chapter/2/next">Next Chapter</a>',
        "html.parser",
    )

    assert (
        extract_next_chapter_link(soup, "https://www.royalroad.com/fiction/1/book/chapter/1/start")
        == "https://www.royalroad.com/fiction/1/book/chapter/2/next"
    )
