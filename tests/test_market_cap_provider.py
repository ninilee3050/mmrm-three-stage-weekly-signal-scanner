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
