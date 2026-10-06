from __future__ import annotations

from market_cap_provider import parse_stockanalysis_market_cap_table


def test_parse_stockanalysis_market_cap_table_extracts_ranked_companies() -> None:
    html = """
    <html>
      <body>
        <table>
          <thead>
            <tr>
              <th>No.</th>
              <th>Symbol</th>
              <th>Company Name</th>
              <th>Market Cap</th>
              <th>Stock Price</th>
              <th>% Change</th>
              <th>Revenue</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>NVDA</td>
              <td>NVIDIA Corporation</td>
              <td>5.34T</td>
              <td>220.61</td>
              <td>-0.77%</td>
              <td>215.94B</td>
            </tr>
            <tr>
              <td>2</td>
              <td>GOOGL</td>
              <td>Alphabet Inc.</td>
              <td>4.70T</td>
              <td>387.66</td>
              <td>-2.34%</td>
              <td>422.50B</td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """

    companies = parse_stockanalysis_market_cap_table(html)

    assert len(companies) == 2
    assert companies[0].rank == 1
    assert companies[0].ticker == "NVDA"
    assert companies[0].company == "NVIDIA Corporation"
    assert companies[0].market_cap == "5.34T"
    assert companies[1].ticker == "GOOGL"


def test_preferred_shares_and_second_listings_are_removed_and_ranks_renumbered() -> None:
    from market_cap_provider import MarketCapCompany, common_stock_listings

    listings = [
        MarketCapCompany(1, "NVDA", "NVIDIA Corporation", "5.59T"),
        MarketCapCompany(2, "BRK.B", "Berkshire Hathaway Inc.", "1.08T"),
        MarketCapCompany(3, "BAC.PRO", "Bank of America Corporation", "396.49B"),
        MarketCapCompany(4, "BAC", "Bank of America Corporation", "390.06B"),
        MarketCapCompany(5, "MS.PRE", "Morgan Stanley", "308.32B"),
        MarketCapCompany(6, "MS", "Morgan Stanley", "304.55B"),
        MarketCapCompany(7, "PBR", "Petrobras", "126.19B"),
        MarketCapCompany(8, "PBR.A", "Petrobras", "126.19B"),
    ]

    kept = common_stock_listings(listings)

    assert [(item.rank, item.ticker) for item in kept] == [
        (1, "NVDA"),
        (2, "BRK.B"),
        (3, "BAC"),
        (4, "MS"),
        (5, "PBR"),
    ]


CURRENT_LAYOUT_HTML = """
<table><thead><tr>
  <th>Rank</th><th>Company</th><th>Market Cap</th><th>Price</th><th>Today</th>
</tr></thead><tbody>
<tr><td>1</td><td><a href="/stocks/nvda/"><div><img alt=""/> <span>N</span></div>
  <div><div>NVIDIA Corporation</div> <div>NVDA</div></div></a></td>
  <td>5.77T</td><td>$238.90</td><td>+2.12%</td></tr>
<tr><td>12</td><td><a href="/stocks/brk.b/"><div><span>B</span></div>
  <div><div>Berkshire Hathaway Inc.</div> <div>BRK.B</div></div></a></td>
  <td>1.08T</td><td>$500.00</td><td>+0.10%</td></tr>
<tr><td>51</td><td><a href="/stocks/gs.prd/"><div><span>T</span></div>
  <div><div>The Goldman Sachs Group, Inc.</div> <div>GS.PRD</div></div></a></td>
  <td>283.22B</td><td>$20.00</td><td>0.00%</td></tr>
<tr><td>60</td><td><a href="/stocks/t/"><div><span>A</span></div>
  <div><div>AT&amp;T Inc.</div> <div>T</div></div></a></td>
  <td>200.00B</td><td>$28.00</td><td>0.00%</td></tr>
</tbody></table>
"""


def test_parse_current_layout_with_combined_company_cell() -> None:
    companies = parse_stockanalysis_market_cap_table(CURRENT_LAYOUT_HTML)

    assert [(c.rank, c.ticker, c.company, c.market_cap) for c in companies] == [
        (1, "NVDA", "NVIDIA Corporation", "5.77T"),
        (12, "BRK.B", "Berkshire Hathaway Inc.", "1.08T"),
        (51, "GS.PRD", "The Goldman Sachs Group, Inc.", "283.22B"),
        (60, "T", "AT&T Inc.", "200.00B"),
    ]


def test_ranking_uses_next_page_to_replace_preferred_shares(monkeypatch) -> None:
    import market_cap_provider as provider

    page_two = CURRENT_LAYOUT_HTML.replace("/stocks/nvda/", "/stocks/aapl/").replace(
        "NVIDIA Corporation", "Apple Inc."
    ).replace("NVDA", "AAPL").replace("<td>1</td>", "<td>101</td>")
    pages = {1: CURRENT_LAYOUT_HTML, 2: page_two, 3: ""}
    requested = []

    def fake_download(page: int = 1) -> str:
        requested.append(page)
        return pages[page]

    monkeypatch.setattr(provider, "_download_stockanalysis_page", fake_download)

    result = provider.fetch_us_top_market_cap_result(limit=4, cache_path=None)

    assert requested == [1, 2]
    assert [(c.rank, c.ticker) for c in result.companies] == [
        (1, "NVDA"),
        (2, "BRK.B"),
        (3, "T"),
        (4, "AAPL"),
    ]
    assert result.warning == ""


def test_saved_ranking_is_used_when_the_live_page_cannot_be_read(
    monkeypatch, tmp_path
) -> None:
    import pytest

    import market_cap_provider as provider

    cache = tmp_path / "top100_cache.json"
    monkeypatch.setattr(
        provider, "_download_stockanalysis_page", lambda page=1: CURRENT_LAYOUT_HTML
    )
    live = provider.fetch_us_top_market_cap_result(limit=3, cache_path=cache)
    assert live.warning == "" and cache.exists()

    # The site changes its layout: nothing can be parsed any more.
    monkeypatch.setattr(
        provider, "_download_stockanalysis_page", lambda page=1: "<html></html>"
    )
    fallback = provider.fetch_us_top_market_cap_result(limit=3, cache_path=cache)

    assert [c.ticker for c in fallback.companies] == ["NVDA", "BRK.B", "T"]
    assert "저장한 목록을 사용합니다" in fallback.warning
    with pytest.raises(provider.MarketCapLoadError):
        provider.fetch_us_top_market_cap_result(limit=3, cache_path=tmp_path / "none.json")
