from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import pipeline

# A tiny export with one of each problem found in sales.csv.
SMALL_EXPORT = """\
Order ID,Region,Product,Quantity ,Unit Price,Discount,Order Date
SO-1,North,Mouse,2,29.50,,2026-01-01
SO-2, north ,Mouse,1 000,"29,50",5.00,2026-01-02
SO-3,EAST,Keyboard,3,,,2026-01-03
SO-4,East,Keyboard,1,89.00,4.45,2026-01-04
Order ID,Region,Product,Quantity ,Unit Price,Discount,Order Date
SO-4,East,Keyboard,1,89.00,4.45,2026-01-04
SO-5,East,Keyboard,-1,89.00,,2026-01-05
TOTAL,,,1006,,9.45,
"""


def text_frame(columns: dict[str, list[str | None]]) -> pd.DataFrame:
    """Build a frame of raw text cells, the way load_sales returns them."""
    return pd.DataFrame(columns, dtype="str")


@pytest.fixture
def small_export(tmp_path: Path) -> Path:
    path = tmp_path / "sales.csv"
    path.write_text(SMALL_EXPORT, encoding="utf-8")
    return path


@pytest.fixture
def clean(small_export: Path) -> pd.DataFrame:
    return pipeline.clean_sales(pipeline.load_sales(small_export))


def test_load_sales_treats_only_empty_cells_as_missing(tmp_path: Path) -> None:
    path = tmp_path / "sales.csv"
    path.write_text("Region,Discount\nNA,\n", encoding="utf-8")

    sales = pipeline.load_sales(path)

    assert sales["Region"].tolist() == ["NA"]
    assert sales["Discount"].isna().tolist() == [True]


def test_drop_repeated_headers() -> None:
    raw = text_frame(
        {"Order ID": ["SO-1", "Order ID", "SO-2"], "Quantity ": ["1", "Quantity ", "2"]}
    )

    result = pipeline.drop_repeated_headers(raw)

    assert result["Order ID"].tolist() == ["SO-1", "SO-2"]


def test_normalize_column_names() -> None:
    raw = pd.DataFrame(columns=["Order ID", "Quantity ", "Unit Price", "Order Date"])

    result = pipeline.normalize_column_names(raw)

    assert result.columns.tolist() == ["order_id", "quantity", "unit_price", "order_date"]


def test_drop_total_rows() -> None:
    raw = text_frame({"order_id": ["SO-1", "TOTAL", " total "]})

    assert pipeline.drop_total_rows(raw)["order_id"].tolist() == ["SO-1"]


def test_normalize_regions() -> None:
    raw = text_frame({"region": [" South", "EAST", "west  ", "North", "central"]})

    result = pipeline.normalize_regions(raw)

    assert result["region"].tolist() == ["South", "East", "West", "North", "Central"]


def test_parse_quantities_removes_thousands_separators_and_keeps_returns() -> None:
    raw = text_frame({"quantity": ["5", "1 762", " 12 ", "-3"]})

    result = pipeline.parse_quantities(raw)

    assert result["quantity"].dtype == "int64"
    assert result["quantity"].tolist() == [5, 1762, 12, -3]


def test_parse_unit_prices_accepts_decimal_commas() -> None:
    raw = text_frame({"unit_price": ["39,90", "21.00", None]})

    result = pipeline.parse_unit_prices(raw)

    expected = pd.Series([39.9, 21.0, np.nan], name="unit_price")
    assert_series_equal(result["unit_price"], expected)


def test_fill_missing_unit_prices_uses_the_most_common_price() -> None:
    sales = pd.DataFrame(
        {
            "product": ["Mouse", "Mouse", "Mouse", "Desk lamp", "Mouse"],
            "unit_price": [29.5, 29.5, 30.0, 39.9, np.nan],
        }
    )

    result = pipeline.fill_missing_unit_prices(sales)

    assert result["unit_price"].tolist() == [29.5, 29.5, 30.0, 39.9, 29.5]


def test_parse_discounts_turns_empty_into_zero() -> None:
    raw = text_frame({"discount": [None, "5.00", "1,50"]})

    result = pipeline.parse_discounts(raw)

    assert result["discount"].tolist() == [0.0, 5.0, 1.5]


def test_parse_order_dates_rejects_other_formats() -> None:
    with pytest.raises(ValueError):
        pipeline.parse_order_dates(text_frame({"order_date": ["14/11/2026"]}))


def test_drop_duplicate_lines() -> None:
    sales = pd.DataFrame({"order_id": ["SO-1", "SO-2", "SO-1"], "quantity": [1, 2, 1]})

    result = pipeline.drop_duplicate_lines(sales)

    assert result["order_id"].tolist() == ["SO-1", "SO-2"]
    assert result.index.tolist() == [0, 1]


def test_clean_sales_fixes_every_problem(clean: pd.DataFrame) -> None:
    assert clean.columns.tolist() == [
        "order_id",
        "region",
        "product",
        "quantity",
        "unit_price",
        "discount",
        "order_date",
    ]
    assert clean["order_id"].tolist() == ["SO-1", "SO-2", "SO-3", "SO-4", "SO-5"]
    assert clean["region"].tolist() == ["North", "North", "East", "East", "East"]
    assert clean["quantity"].tolist() == [2, 1000, 3, 1, -1]
    assert clean["unit_price"].tolist() == [29.5, 29.5, 89.0, 89.0, 89.0]
    assert clean["discount"].tolist() == [0.0, 5.0, 0.0, 4.45, 0.0]


def test_check_clean_accepts_clean_data(clean: pd.DataFrame) -> None:
    assert pipeline.check_clean(clean) is clean


def test_check_clean_rejects_missing_values(clean: pd.DataFrame) -> None:
    clean.loc[0, "region"] = np.nan

    with pytest.raises(ValueError, match="missing values in region"):
        pipeline.check_clean(clean)


def test_check_clean_rejects_malformed_order_ids(clean: pd.DataFrame) -> None:
    clean.loc[0, "order_id"] = "12345"

    with pytest.raises(ValueError, match="malformed order ids"):
        pipeline.check_clean(clean)


def test_check_clean_rejects_order_ids_used_twice(clean: pd.DataFrame) -> None:
    clean.loc[1, "order_id"] = "SO-1"

    with pytest.raises(ValueError, match="order ids shared by different lines"):
        pipeline.check_clean(clean)


@pytest.mark.parametrize(
    "line",
    [
        "SO-6,West,Mouse,many,29.50,,2026-01-06",
        "SO-6,West,Mouse,1,cheap,,2026-01-06",
        "SO-6,West,Mouse,1,29.50,some,2026-01-06",
        "SO-6,West,Mouse,1,29.50,,06/01/2026",
    ],
)
def test_value_that_cannot_be_parsed_stops_the_pipeline(tmp_path: Path, line: str) -> None:
    path = tmp_path / "sales.csv"
    path.write_text(SMALL_EXPORT + line + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        pipeline.clean_sales(pipeline.load_sales(path))


def test_product_without_any_price_is_not_guessed(tmp_path: Path) -> None:
    path = tmp_path / "sales.csv"
    path.write_text(SMALL_EXPORT + "SO-6,West,Webcam,1,,,2026-01-06\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing values in unit_price"):
        pipeline.clean_sales(pipeline.load_sales(path))


def test_add_revenue(clean: pd.DataFrame) -> None:
    result = pipeline.add_revenue(clean)

    assert result["revenue"].round(2).tolist() == [59.0, 29495.0, 267.0, 84.55, -89.0]


def test_run_writes_revenue_by_region(small_export: Path, tmp_path: Path) -> None:
    target = tmp_path / "revenue_by_region.csv"

    pipeline.run(small_export, target)

    assert target.read_text(encoding="utf-8") == "region,revenue\nEast,262.55\nNorth,29554.00\n"


@pytest.fixture(scope="module")
def real_sales() -> pd.DataFrame:
    return pipeline.clean_sales(pipeline.load_sales(pipeline.SALES_CSV))


def test_real_export_cleans_to_unique_order_lines(real_sales: pd.DataFrame) -> None:
    assert len(real_sales) == 1500
    assert sorted(real_sales["order_id"]) == [f"SO-{n}" for n in range(10001, 11501)]
    assert sorted(real_sales["region"].unique()) == ["Central", "East", "North", "South", "West"]
    assert not real_sales.isna().any().any()


def test_real_total_footer_agrees_with_how_values_are_read() -> None:
    raw = pipeline.drop_repeated_headers(pipeline.load_sales(pipeline.SALES_CSV))
    raw = pipeline.normalize_column_names(raw)
    footer = raw[raw["order_id"] == "TOTAL"]

    lines = pipeline.drop_total_rows(raw).pipe(pipeline.parse_quantities)
    lines = lines.pipe(pipeline.parse_discounts)

    assert lines["quantity"].sum() == int(footer["quantity"].iloc[0])
    assert lines["discount"].sum() == pytest.approx(float(footer["discount"].iloc[0]))


def test_real_order_ids_repeat_only_on_exact_copies() -> None:
    raw = pipeline.drop_repeated_headers(pipeline.load_sales(pipeline.SALES_CSV))
    lines = pipeline.drop_total_rows(pipeline.normalize_column_names(raw))

    repeated_ids = lines["order_id"].duplicated(keep=False)

    assert repeated_ids.sum() == 10
    assert lines[repeated_ids].duplicated().sum() == 5


def test_committed_report_is_up_to_date(tmp_path: Path) -> None:
    target = tmp_path / "revenue_by_region.csv"

    pipeline.run(pipeline.SALES_CSV, target)

    assert target.read_text(encoding="utf-8") == pipeline.REVENUE_CSV.read_text(encoding="utf-8")
