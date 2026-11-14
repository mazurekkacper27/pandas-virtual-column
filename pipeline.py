"""Clean the raw sales export and report the revenue of each region.

Run it with ``uv run python pipeline.py``. It reads ``sales.csv`` next to this file
and writes ``revenue_by_region.csv``. Each fix logs how many rows it touched, so
the output doubles as a short data-quality report.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from solution import add_virtual_column

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
SALES_CSV = HERE / "sales.csv"
REVENUE_CSV = HERE / "revenue_by_region.csv"

REVENUE_ROLE = "quantity * unit_price - discount"
ORDER_ID = r"SO-\d+"


def load_sales(path: Path) -> pd.DataFrame:
    """Read every value as text, so each column is parsed explicitly later on.

    Only empty cells count as missing; strings like "NA" stay as they are.
    """
    sales = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    logger.info("Read %d lines from %s", len(sales), path.name)
    return sales


def drop_repeated_headers(sales: pd.DataFrame) -> pd.DataFrame:
    """Drop copies of the header row that ended up between the data lines."""
    is_header = sales.eq(sales.columns).all(axis=1)
    logger.info("Dropped %d repeated header row(s)", is_header.sum())
    return sales[~is_header]


def normalize_column_names(sales: pd.DataFrame) -> pd.DataFrame:
    """Turn headers like "Quantity " and "Unit Price" into "quantity" and "unit_price".

    add_virtual_column only accepts labels made of letters and underscores, so the
    spaces have to go. Lowercasing isn't required, but it keeps the names stable if
    a later export changes the case of a header.
    """
    return sales.rename(columns=lambda name: re.sub(r"\s+", "_", str(name).strip()).lower())


def drop_total_rows(sales: pd.DataFrame) -> pd.DataFrame:
    """Drop the TOTAL footer of the export. It is a sum, not an order line."""
    is_total = sales["order_id"].str.strip().str.upper().eq("TOTAL")
    logger.info("Dropped %d TOTAL row(s)", is_total.sum())
    return sales[~is_total]


def normalize_regions(sales: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces and fix the case, so " south  " and "SOUTH" both become "South"."""
    regions = sales["region"].str.strip().str.title()
    changed = sales["region"].notna() & regions.ne(sales["region"])
    logger.info("Normalized the region in %d row(s)", changed.sum())
    return sales.assign(region=regions)


def parse_quantities(sales: pd.DataFrame) -> pd.DataFrame:
    """Remove thousands separators ("1 762" becomes 1762) and convert to integers.

    Negative quantities are returns. They stay negative, so they reduce revenue.
    """
    text = sales["quantity"].str.strip()
    has_separator = text.str.contains(r"\d\s+\d", na=False)
    quantities = pd.to_numeric(text.str.replace(r"\s+", "", regex=True)).astype("int64")
    logger.info("Removed a thousands separator from the quantity in %d row(s)", has_separator.sum())
    logger.info("Kept %d row(s) with a negative quantity as returns", quantities.lt(0).sum())
    return sales.assign(quantity=quantities)


def to_amount(values: pd.Series[str]) -> pd.Series[float]:
    """Parse money written with a decimal dot ("39.90") or a decimal comma ("39,90")."""
    text = values.str.strip()
    has_comma = text.str.contains(",", regex=False, na=False)
    logger.info("Read a decimal comma in %s in %d row(s)", values.name, has_comma.sum())
    return pd.to_numeric(text.str.replace(",", ".", regex=False)).astype("float64")


def parse_unit_prices(sales: pd.DataFrame) -> pd.DataFrame:
    return sales.assign(unit_price=to_amount(sales["unit_price"]))


def fill_missing_unit_prices(sales: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing unit price with the list price of the product.

    The list price is the price the product is most often sold at. In this export
    every product is sold at exactly one price, so the filled value matches all the
    other lines of that product.
    """
    list_prices = (
        sales.dropna(subset=["unit_price"])
        .groupby("product")["unit_price"]
        .agg(lambda prices: prices.mode().iloc[0])
    )
    missing = sales["unit_price"].isna()
    logger.info("Filled the unit price from the list price in %d row(s)", missing.sum())
    return sales.assign(unit_price=sales["unit_price"].fillna(sales["product"].map(list_prices)))


def parse_discounts(sales: pd.DataFrame) -> pd.DataFrame:
    """An empty discount means there is no discount, so it becomes 0."""
    discounts = to_amount(sales["discount"])
    logger.info("Set an empty discount to 0 in %d row(s)", discounts.isna().sum())
    return sales.assign(discount=discounts.fillna(0.0))


def parse_order_dates(sales: pd.DataFrame) -> pd.DataFrame:
    """Dates must be YYYY-MM-DD. Anything else stops the pipeline instead of being guessed."""
    dates = pd.to_datetime(sales["order_date"].str.strip(), format="%Y-%m-%d")
    return sales.assign(order_date=dates)


def drop_duplicate_lines(sales: pd.DataFrame) -> pd.DataFrame:
    """Drop order lines that were exported twice.

    It runs after the values are normalized, so it also catches copies that only
    differed in formatting.
    """
    is_copy = sales.duplicated()
    logger.info("Dropped %d duplicated line(s)", is_copy.sum())
    return sales[~is_copy].reset_index(drop=True)


def check_clean(sales: pd.DataFrame) -> pd.DataFrame:
    """Stop on what the steps above couldn't fix: gaps, malformed or repeated order ids."""
    problems = []
    if columns_with_gaps := [str(name) for name in sales.columns[sales.isna().any()]]:
        problems.append(f"missing values in {', '.join(columns_with_gaps)}")
    if not sales["order_id"].str.fullmatch(ORDER_ID).all():
        problems.append("malformed order ids")
    if sales["order_id"].duplicated().any():
        problems.append("order ids shared by different lines")
    if problems:
        raise ValueError("Sales data is still not clean: " + "; ".join(problems))
    return sales


def clean_sales(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.pipe(drop_repeated_headers)
        .pipe(normalize_column_names)
        .pipe(drop_total_rows)
        .pipe(normalize_regions)
        .pipe(parse_quantities)
        .pipe(parse_unit_prices)
        .pipe(fill_missing_unit_prices)
        .pipe(parse_discounts)
        .pipe(parse_order_dates)
        .pipe(drop_duplicate_lines)
        .pipe(check_clean)
    )


def add_revenue(sales: pd.DataFrame) -> pd.DataFrame:
    with_revenue = add_virtual_column(sales, REVENUE_ROLE, "revenue")
    if "revenue" not in with_revenue.columns:
        raise ValueError(f"Cannot compute revenue as {REVENUE_ROLE!r}")
    return with_revenue


def revenue_by_region(sales: pd.DataFrame) -> pd.DataFrame:
    return sales.groupby("region")["revenue"].sum().round(2).reset_index()


def run(source: Path = SALES_CSV, target: Path = REVENUE_CSV) -> pd.DataFrame:
    sales = clean_sales(load_sales(source)).pipe(add_revenue)
    report = revenue_by_region(sales)
    report.to_csv(target, index=False, float_format="%.2f", lineterminator="\n")
    logger.info("Wrote the revenue of %d regions to %s", len(report), target.name)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()


if __name__ == "__main__":
    main()
