import pytest

BeautifulSoup = pytest.importorskip("bs4").BeautifulSoup

from novel_memory.scraper import extract_chapter_text, extract_next_chapter_link


def test_extract_chapter_text_from_royalroad_markup():
    soup = BeautifulSoup(
        """
        <html><body>
          <div class="chapter-inner chapter-content">
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
          </div>
        </body></html>
        """,
        "html.parser",
    )

    assert extract_chapter_text(soup) == "First paragraph.\nSecond paragraph."


def test_extract_next_chapter_link_makes_absolute_url():
    soup = BeautifulSoup(
        '<a href="/fiction/1/book/chapter/2/next">Next Chapter</a>',
        "html.parser",
    )

    assert (
        extract_next_chapter_link(soup, "https://www.royalroad.com/fiction/1/book/chapter/1/start")
        == "https://www.royalroad.com/fiction/1/book/chapter/2/next"
    )
