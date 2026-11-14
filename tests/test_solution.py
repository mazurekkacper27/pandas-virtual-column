import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from solution import add_virtual_column


@pytest.fixture
def numbers() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})


def test_adds_the_column_after_the_existing_ones(numbers: pd.DataFrame) -> None:
    result = add_virtual_column(numbers, "a + b", "total")

    expected = numbers.assign(total=[5, 7, 9])
    assert_frame_equal(result, expected)


def test_example_from_the_task() -> None:
    fruits_sales = pd.DataFrame(
        {"name": ["banana", "apple"], "quantity": [10, 3], "price": [10, 1]}
    )

    result = add_virtual_column(fruits_sales, "quantity * price", "total")

    assert result["total"].tolist() == [100, 3]


def test_multiplication_binds_tighter_than_addition(numbers: pd.DataFrame) -> None:
    result = add_virtual_column(numbers, "a + b * c", "d")

    assert result["d"].tolist() == [29, 42, 57]


def test_subtraction_is_applied_left_to_right(numbers: pd.DataFrame) -> None:
    result = add_virtual_column(numbers, "a - b - c", "d")

    assert result["d"].tolist() == [-10, -11, -12]


def test_mixed_operations(numbers: pd.DataFrame) -> None:
    result = add_virtual_column(numbers, "c - a * b + a * a * c", "d")

    assert result["d"].tolist() == [7 - 4 + 7, 8 - 10 + 32, 9 - 18 + 81]


def test_single_column_is_copied(numbers: pd.DataFrame) -> None:
    result = add_virtual_column(numbers, "b", "d")

    assert result["d"].tolist() == [4, 5, 6]


def test_long_role(numbers: pd.DataFrame) -> None:
    result = add_virtual_column(numbers, " + ".join(["a"] * 100), "d")

    assert result["d"].tolist() == [100, 200, 300]


def test_label_can_be_just_an_underscore() -> None:
    df = pd.DataFrame({"_": [3], "a": [2]})

    result = add_virtual_column(df, "_*a", "b")

    assert result["b"].tolist() == [6]


def test_works_with_floats() -> None:
    lines = pd.DataFrame({"quantity": [2, 3], "unit_price": [9.5, 0.25], "discount": [1.0, 0.0]})

    result = add_virtual_column(lines, "quantity * unit_price - discount", "revenue")

    assert result["revenue"].tolist() == [18.0, 0.75]


@pytest.mark.parametrize("role", ["a+b", "  a +   b  ", "a\t+\nb"])
def test_whitespace_does_not_matter(numbers: pd.DataFrame, role: str) -> None:
    result = add_virtual_column(numbers, role, "d")

    assert result["d"].tolist() == [5, 7, 9]


def test_does_not_modify_the_input(numbers: pd.DataFrame) -> None:
    original = numbers.copy()

    result = add_virtual_column(numbers, "a * b", "d")

    assert result is not numbers
    assert_frame_equal(numbers, original)


def test_keeps_the_index() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["first", "second"])

    result = add_virtual_column(df, "a * b", "c")

    assert result.index.tolist() == ["first", "second"]
    assert result["c"].tolist() == [3, 8]


def test_labels_may_use_any_letters() -> None:
    df = pd.DataFrame({"cena": [2.5], "ilość": [4], "Rabat": [1.0]})

    result = add_virtual_column(df, "cena * ilość - Rabat", "Wartość_netto")

    assert result["Wartość_netto"].tolist() == [9.0]


def test_existing_column_name_gives_an_empty_frame(numbers: pd.DataFrame) -> None:
    assert add_virtual_column(numbers, "a + b", "a").empty


@pytest.mark.parametrize("label", ["Order Date", "b1", "b-c", 0])
def test_invalid_label_of_any_column_gives_an_empty_frame(label: str | int) -> None:
    df = pd.DataFrame([[1, 2]], columns=pd.Index(["a", label]))

    assert add_virtual_column(df, "a + a", "c").empty


def test_ambiguous_column_gives_an_empty_frame(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])

    assert add_virtual_column(df, "a + b", "c").empty
    assert "there is more than one column 'a'" in caplog.text


@pytest.mark.parametrize(
    "role",
    [
        "",
        "   ",
        "a +",
        "+ a",
        "-a",
        "a + + b",
        "a b",
        "a / b",
        "a ** b",
        "(a + b)",
        "a + 2",
        "a + d",
        "a + label3",
        "a + label&one",
        "a + b_",
        "+",
        "*",
        "a - - b",
        "a * * b",
        "a.b",
        "a, b",
        "a + b;",
        "a = b",
    ],
)
def test_invalid_role_gives_an_empty_frame(numbers: pd.DataFrame, role: str) -> None:
    assert add_virtual_column(numbers, role, "d").empty


@pytest.mark.parametrize("new_column", ["", "d1", "new col", "new-col", " d", "a"])
def test_invalid_new_column_gives_an_empty_frame(numbers: pd.DataFrame, new_column: str) -> None:
    assert add_virtual_column(numbers, "a + b", new_column).empty


@pytest.mark.parametrize("column", [["x", "y"], [True, False]])
def test_non_numeric_column_gives_an_empty_frame(column: list[object]) -> None:
    df = pd.DataFrame({"a": [1, 2], "other": column})

    assert add_virtual_column(df, "a + other", "d").empty


def test_invalid_input_is_logged(numbers: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    add_virtual_column(numbers, "a + d", "e")

    assert "there is no column 'd'" in caplog.text
