# pandas virtual column

`add_virtual_column` adds a column computed from a rule like `"quantity * price"` to a pandas
DataFrame. `pipeline.py` uses it to clean a raw sales export and report revenue per region.

## Setup

You need [uv](https://docs.astral.sh/uv/). Then:

```sh
uv sync
```

This installs Python 3.13 if it's missing, plus pandas and the dev tools (pytest, mypy,
pandas-stubs, ruff) into `.venv`.

## Running

```sh
uv run python pipeline.py   # reads sales.csv, writes revenue_by_region.csv
uv run pytest               # every test, the provided ones included
uv run mypy                 # strict type check
uv run ruff check .         # lint
```

The pipeline logs how many rows each fix touched:

```text
Read 1507 lines from sales.csv
Dropped 1 repeated header row(s)
Dropped 1 TOTAL row(s)
Normalized the region in 15 row(s)
Removed a thousands separator from the quantity in 4 row(s)
Kept 6 row(s) with a negative quantity as returns
Read a decimal comma in unit_price in 7 row(s)
Filled the unit price from the list price in 4 row(s)
Read a decimal comma in discount in 0 row(s)
Set an empty discount to 0 in 963 row(s)
Dropped 5 duplicated line(s)
Wrote the revenue of 5 regions to revenue_by_region.csv
```

## Layout

| File | What it is |
|---|---|
| `solution.py` | `add_virtual_column` and the small parser behind it |
| `pipeline.py` | cleaning steps for `sales.csv` and the revenue report |
| `revenue_by_region.csv` | output of `pipeline.py` |
| `tests/test_solution.py` | tests for `add_virtual_column` |
| `tests/test_pipeline.py` | tests for each cleaning step, an end-to-end run and checks on the real file |
| `test_virtual_column.py`, `sales.csv` | provided with the task, unchanged |

`test_virtual_column.py` is kept exactly as delivered. It has no type annotations and doesn't
follow the lint and format rules (long lines, import order), and changing it isn't allowed. So
mypy and ruff skip it, but pytest runs it with everything else.

## How `add_virtual_column` works

```python
>>> add_virtual_column(fruits_sales, "quantity * price", "total")
     name  quantity  price  total
0  banana        10     10    100
1   apple         3      1      3
```

The role goes through a tokenizer and a small recursive-descent parser. The parser builds a tree
of `Column` and `BinaryOperation` nodes, and each node evaluates itself on a DataFrame. This is
the Interpreter pattern. The role never reaches `eval` or `DataFrame.eval`, so it can only add,
subtract and multiply columns. When the role or a column label is incorrect, the label check,
the tokenizer, the parser or a `Column` node raises `InvalidRoleError`. `add_virtual_column`
catches it, logs the reason as a warning and returns an empty DataFrame, as the task asks.

## Assumptions

### `add_virtual_column`

- In the task's example the new column shows up as `price_total` although `new_column` is
  `"total"`. The task defines `new_column` as the name of the new column, and the provided tests
  expect exactly that name, so I treated `price_total` as a typo.
- The task asks for `+`, `-` and `*`. Apart from column labels and whitespace, these are the only
  things a role can contain. `*` binds tighter than `+` and `-`, as usual in maths. Operators of
  the same precedence are applied left to right, so `a - b - c` is `(a - b) - c`.
- Anything else makes the role incorrect and gives an empty DataFrame: parentheses, unary minus,
  other operators (`/`, `**`, ...) and numbers. A number is not a column label, because labels
  consist only of letters and underscores.
- "Letters" are any letters, not only `A-Z`. So `ilość` is a valid label and `b1` is not. A label
  must have at least one character.
- The task says the result is empty if any column label is incorrect, so every label is checked:
  the columns of `df`, the labels in the role and `new_column`. A DataFrame with a column like
  `Order Date` gives an empty result even if the role doesn't use it. That's why the pipeline
  renames the headers before it adds `revenue`.
- `new_column` must not exist in `df` yet. The task asks for the original data along with an
  additional column, and reusing a name would overwrite data instead. So an existing name counts
  as an incorrect label.
- The role is a mathematical expression, so the columns it uses must exist and hold numbers. On
  text, `+` would join strings, and `*` with a number would repeat them. On booleans, `+` is a
  logical or and `-` fails. A role that uses such a column is incorrect, and so is a label that
  matches more than one column.
- The input DataFrame is never modified. The result is a new DataFrame and keeps its index.

### Pipeline

- Column labels only need letters and underscores, so capitals would be fine. Only the spaces
  have to go. I still lowercase the headers as well (`Unit Price` becomes `unit_price`). The
  revenue rule refers to the columns by fixed names, and this way it keeps working if a later
  export changes the case of a header.
- Each line of `sales.csv` is one order line, and the order ID identifies it. The IDs run from
  SO-10001 to SO-11500 without gaps. Each appears once, except for five that appear twice with
  identical values, so those lines were exported twice.
- The task says not to drop rows that can be fixed. The pipeline only drops rows that aren't order
  lines (the repeated header and the `TOTAL` footer) and the second copy of a duplicated line.
  Every other row is kept, and fixed where needed.
- A negative quantity is a return. It stays negative, so it lowers the region's revenue.
- `1 762` is 1762 with a space as the thousands separator. These four quantities are much larger
  than the other positive quantities (1 to 40), so they look like bulk orders. I kept them, but
  they would be worth confirming with whoever owns the data, because they add about 324k to the
  totals.
- A missing unit price is the product's list price. That's the price the product is most often
  sold at, and in this file every product has exactly one price.
- Revenue isn't rounded per line. Region totals are rounded to cents and sorted by region name.
- The order date isn't needed for the report, but it is still parsed and checked.
- Some problems can't be fixed without guessing, so they stop the pipeline with an error: a
  quantity, price, discount or date that can't be parsed, a value still missing after cleaning
  (such as the price of a product that never has one), a malformed order ID, or one order ID on
  different lines. That's safer than changing the totals without anyone noticing. It doesn't
  catch everything, though. A misspelled region, for example, would get through.

## Problems found in `sales.csv`

The file has 1507 lines under the header. Line numbers count the header as line 1.

| Problem | Rows | How it's handled |
|---|---|---|
| Header `Quantity ` ends with a space, and `Order ID`, `Unit Price`, `Order Date` contain spaces, which `add_virtual_column` labels can't contain | 1 (the header line) | Trimmed, spaces turned into `_` and lowercased (see assumptions): `order_id`, `region`, `product`, `quantity`, `unit_price`, `discount`, `order_date` |
| The header row is repeated in the middle of the file (line 753) | 1 | Dropped |
| A `TOTAL` footer row (line 1508) | 1 | Dropped, it's a sum and not an order line |
| Region with extra spaces or the wrong case: ` South`, `EAST`, `west  `, `central`, ... | 15 | Trimmed and title-cased, which leaves the five regions Central, East, North, South, West |
| Quantity with a space as the thousands separator: `1 762`, `1 745`, `1 371`, `1 427` (SO-10251, SO-10734, SO-11088, SO-11312) | 4 | Separator removed |
| Negative quantity, -2 to -5 (SO-10421, SO-10515, SO-10779, SO-11011, SO-11289, SO-11445) | 6 | Kept as returns |
| Unit price with a decimal comma, like `"39,90"` (SO-10137, SO-10402, SO-10688, SO-10911, SO-11203, SO-11377, SO-11460) | 7 | Comma read as the decimal point |
| Missing unit price (SO-10610, SO-11099 Printer paper; SO-11350, SO-11477 Keyboard) | 4 | Filled with the product's list price (21.00 and 89.00) |
| Lines exported twice (SO-10305, SO-10822, SO-10966, SO-11240, SO-11405) | 5 | Second copy dropped |

That leaves 1500 order lines. Products and dates were already clean: 8 product names and valid
`YYYY-MM-DD` dates from 2026-01-02 to 2026-09-29.

One thing isn't a problem but still needs handling: 963 lines have an empty discount. The task
says an empty discount means there is no discount, so the pipeline sets it to 0 before
subtracting it. Every discount that is filled in is 5% or 10% of its line's gross value, rounded
to the cent. That agrees with the task: the discount is an amount for the whole line.

Some checks back up the decisions above:

- The `TOTAL` footer matches the sums of the raw lines: discount 65924.67, and quantity 37150
  when `1 762` is read as 1762 and negative quantities stay negative. It was clearly computed
  before deduplication. So the export reads these values the same way I do. A test checks both sums.
- SO-11477 has no unit price but a discount of 164.65, which is 5% of 37 × 89.00. That matches
  the filled Keyboard price.

## Result

`revenue_by_region.csv`:

| region | revenue |
|---|---:|
| Central | 515824.67 |
| East | 442219.91 |
| North | 665897.52 |
| South | 504036.04 |
| West | 442268.89 |
