"""Add a computed ("virtual") column to a pandas DataFrame.

A role is a small arithmetic expression over column labels, for example
``"quantity * unit_price - discount"``. It is parsed into a tree of nodes that
evaluate themselves against a DataFrame, so the role never goes through ``eval``.

Grammar::

    expression := term (("+" | "-") term)*
    term       := column ("*" column)*
    column     := a label made only of letters and underscores
"""

from __future__ import annotations

import logging
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas
from pandas.api.types import is_bool_dtype, is_numeric_dtype

logger = logging.getLogger(__name__)

TOKEN = re.compile(r"(?P<word>\w+)|(?P<operator>[-+*])|(?P<space>\s+)|(?P<invalid>.)")

type Values = pandas.Series[float]

OPERATIONS: dict[str, Callable[[Values, Values], Values]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}


class InvalidRoleError(ValueError):
    """The role or a column label is incorrect."""


def is_label(label: object) -> bool:
    """A column label consists only of letters and underscores, like ``unit_price``."""
    return (
        isinstance(label, str)
        and label != ""
        and all(char == "_" or char.isalpha() for char in label)
    )


class Expression(Protocol):
    def evaluate(self, df: pandas.DataFrame) -> Values: ...


@dataclass(frozen=True)
class Column:
    label: str

    def evaluate(self, df: pandas.DataFrame) -> Values:
        matches = list(df.columns).count(self.label)
        if matches == 0:
            raise InvalidRoleError(f"there is no column {self.label!r}")
        if matches > 1:
            raise InvalidRoleError(f"there is more than one column {self.label!r}")
        values = df[self.label]
        # The role is a mathematical expression, so it only makes sense on numbers.
        if is_bool_dtype(values) or not is_numeric_dtype(values):
            raise InvalidRoleError(f"column {self.label!r} is not numeric")
        return values


@dataclass(frozen=True)
class BinaryOperation:
    symbol: str
    left: Expression
    right: Expression

    def evaluate(self, df: pandas.DataFrame) -> Values:
        apply = OPERATIONS[self.symbol]
        return apply(self.left.evaluate(df), self.right.evaluate(df))


def tokenize(role: str) -> list[str]:
    tokens = []
    for match in TOKEN.finditer(role):
        if match.lastgroup == "invalid":
            raise InvalidRoleError(f"unexpected character {match.group()!r}")
        if match.lastgroup != "space":
            tokens.append(match.group())
    return tokens


class Parser:
    """Recursive-descent parser: ``*`` binds tighter than ``+`` and ``-``,
    and operators of the same precedence are applied from left to right."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._position = 0

    def parse(self) -> Expression:
        expression = self._expression()
        if (leftover := self._peek()) is not None:
            raise InvalidRoleError(f"unexpected {leftover!r}")
        return expression

    def _expression(self) -> Expression:
        node = self._term()
        while (symbol := self._peek()) in ("+", "-"):
            self._position += 1
            node = BinaryOperation(symbol, node, self._term())
        return node

    def _term(self) -> Expression:
        node = self._column()
        while (symbol := self._peek()) == "*":
            self._position += 1
            node = BinaryOperation(symbol, node, self._column())
        return node

    def _column(self) -> Expression:
        token = self._peek()
        if token is None or not is_label(token):
            raise InvalidRoleError(f"expected a column label, got {token!r}")
        self._position += 1
        return Column(token)

    def _peek(self) -> str | None:
        if self._position < len(self._tokens):
            return self._tokens[self._position]
        return None


def parse_role(role: str) -> Expression:
    return Parser(tokenize(role)).parse()


def check_labels(df: pandas.DataFrame, new_column: str) -> None:
    for label in [*df.columns, new_column]:
        if not is_label(label):
            raise InvalidRoleError(f"{label!r} is not a valid column label")
    if new_column in df.columns:
        raise InvalidRoleError(f"column {new_column!r} already exists")


def add_virtual_column(df: pandas.DataFrame, role: str, new_column: str) -> pandas.DataFrame:
    """Return a new DataFrame with the data of ``df`` and an additional ``new_column``.

    ``role`` adds, subtracts and multiplies numeric columns, for example
    ``"quantity * price"``. An empty DataFrame is returned when the role is
    incorrect, when any column label (of ``df``, in the role or ``new_column``)
    is not made only of letters and underscores, or when ``new_column`` exists.
    """
    try:
        check_labels(df, new_column)
        values = parse_role(role).evaluate(df)
    except InvalidRoleError as error:
        logger.warning("Cannot add column %r: %s", new_column, error)
        return pandas.DataFrame([])
    return df.assign(**{new_column: values})
