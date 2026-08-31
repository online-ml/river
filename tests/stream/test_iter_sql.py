from __future__ import annotations

import datetime as dt
import gc
import json
import os
import typing
import zipfile

import pytest
import sqlalchemy
from sqlalchemy import orm

from river import stream
from river.base.typing import FeatureName

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterator

Executor = sqlalchemy.Connection | orm.Session
"""What `iter_sql` accepts as its `conn` argument."""

Row = tuple[dict[FeatureName, typing.Any], typing.Any]
"""One `(x, y)` pair, as `iter_sql` yields it."""

METADATA = sqlalchemy.MetaData()
T_SALES = sqlalchemy.Table(
    "sales",
    METADATA,
    sqlalchemy.Column("shop", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column("date", sqlalchemy.Date, primary_key=True),
    sqlalchemy.Column("amount", sqlalchemy.Integer),
)

SALES: list[dict[str, typing.Any]] = [
    {"shop": "Hema", "date": dt.date(2016, 8, 2), "amount": 20},
    {"shop": "Ikea", "date": dt.date(2016, 8, 2), "amount": 18},
    {"shop": "Hema", "date": dt.date(2016, 8, 3), "amount": 22},
]

# `date` is left out of the query forms on purpose: whether it comes back as a `dt.date` or as the
# text SQLite stores it as depends on the form, which one test covers on its own below.
SHOPS_AND_AMOUNTS = "SELECT shop, amount FROM sales"
INSERT = T_SALES.insert().values(shop="Zeeman", date=dt.date(2016, 8, 5), amount=9)


QUERY_CASES: dict[str, str | sqlalchemy.Executable] = {
    "raw-string": SHOPS_AND_AMOUNTS,
    "text-clause": sqlalchemy.text(SHOPS_AND_AMOUNTS),
    "select": sqlalchemy.select(T_SALES.c.shop, T_SALES.c.amount),
    "compound-select": sqlalchemy.union_all(sqlalchemy.select(T_SALES.c.shop, T_SALES.c.amount)),
}

TARGET_CASES: dict[str, tuple[str | None, list[Row]]] = {
    "no-target": (
        None,
        [
            ({"shop": "Hema", "amount": 20}, None),
            ({"shop": "Ikea", "amount": 18}, None),
            ({"shop": "Hema", "amount": 22}, None),
        ],
    ),
    "amount-as-target": (
        "amount",
        [({"shop": "Hema"}, 20), ({"shop": "Ikea"}, 18), ({"shop": "Hema"}, 22)],
    ),
}

COLUMN_CASES: dict[str, tuple[str, list[Row]]] = {
    "nulls-stay-none": (
        "SELECT shop, NULL AS bonus FROM sales LIMIT 1",
        [({"shop": "Hema", "bonus": None}, None)],
    ),
    "a-duplicated-label-keeps-the-last-value": (
        "SELECT shop AS v, amount AS v FROM sales LIMIT 1",
        [({"v": 20}, None)],
    ),
    "no-matching-row-yields-nothing": ("SELECT shop FROM sales WHERE shop = 'Zeeman'", []),
}

ERROR_CASES: dict[str, tuple[str | sqlalchemy.Executable, str | None, type[Exception]]] = {
    "an-unknown-target-name": ("SELECT shop FROM sales", "amount", KeyError),
    "a-statement-returning-no-rows": (INSERT, None, sqlalchemy.exc.ResourceClosedError),
}

# Both accept an `Executable` and return a `Result`, so both are valid `conn` arguments. An
# `Engine` is not: `Engine.execute` was removed in SQLAlchemy 2.0.
EXECUTOR_CASES: dict[str, Callable[[sqlalchemy.Engine], Executor]] = {
    "connection": sqlalchemy.Engine.connect,
    "session": orm.Session,
}


@pytest.fixture
def engine() -> sqlalchemy.Engine:
    engine = sqlalchemy.create_engine("sqlite://")  # in-memory
    METADATA.create_all(engine)
    with engine.connect() as conn:
        _ = conn.execute(T_SALES.insert(), SALES)
        conn.commit()
    return engine


@pytest.fixture(params=EXECUTOR_CASES.values(), ids=list(EXECUTOR_CASES))
def executor(engine: sqlalchemy.Engine, request: pytest.FixtureRequest) -> Iterator[Executor]:
    """A connection and a session, so that every query runs against both `execute` overloads."""
    open_executor = typing.cast("Callable[[sqlalchemy.Engine], Executor]", request.param)
    conn = open_executor(engine)
    try:
        yield conn
    finally:
        conn.close()


@pytest.mark.parametrize("query", QUERY_CASES.values(), ids=QUERY_CASES)
@pytest.mark.parametrize(("target_name", "expected"), TARGET_CASES.values(), ids=TARGET_CASES)
def test_a_query_is_iterated_row_by_row(
    executor: Executor,
    query: str | sqlalchemy.Executable,
    target_name: str | None,
    expected: list[Row],
) -> None:
    assert list(stream.iter_sql(query, executor, target_name=target_name)) == expected


@pytest.mark.parametrize(("query", "expected"), COLUMN_CASES.values(), ids=COLUMN_CASES)
def test_columns_are_mapped_to_features(
    executor: Executor, query: str, expected: list[Row]
) -> None:
    assert list(stream.iter_sql(query, executor)) == expected


@pytest.mark.parametrize(("query", "target_name", "error"), ERROR_CASES.values(), ids=ERROR_CASES)
def test_iterating_raises(
    executor: Executor,
    query: str | sqlalchemy.Executable,
    target_name: str | None,
    error: type[Exception],
) -> None:
    dataset = stream.iter_sql(query, executor, target_name=target_name)

    with pytest.raises(error):
        _ = next(dataset)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (sqlalchemy.select(T_SALES), dt.date(2016, 8, 2)),
        ("SELECT shop, date, amount FROM sales", "2016-08-02"),
    ],
    ids=["select", "raw-string"],
)
def test_dates_are_only_parsed_when_the_query_carries_the_schema(
    executor: Executor, query: str | sqlalchemy.Executable, expected: typing.Any
) -> None:
    """A raw string gives SQLAlchemy no column types, so SQLite hands back the stored text."""
    x, _ = next(stream.iter_sql(query, executor))

    assert x["date"] == expected


def test_a_returning_statement_is_iterated(executor: Executor) -> None:
    """A non-`SELECT` is a valid query as long as it returns rows."""
    query = INSERT.returning(T_SALES.c.shop, T_SALES.c.amount)

    assert list(stream.iter_sql(query, executor, target_name="amount")) == [({"shop": "Zeeman"}, 9)]


@pytest.fixture
def executed_results(
    executor: Executor, monkeypatch: pytest.MonkeyPatch
) -> list[sqlalchemy.Result[typing.Any]]:
    """Records the results `iter_sql` gets back, so tests can check it closes them."""
    results: list[sqlalchemy.Result[typing.Any]] = []
    execute = executor.execute

    def spy(*args: typing.Any, **kwargs: typing.Any) -> sqlalchemy.Result[typing.Any]:
        result = execute(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(executor, "execute", spy)
    return results


@pytest.mark.parametrize("exhaust", [True, False], ids=["exhausted", "abandoned"])
def test_the_result_is_closed(
    executor: Executor, executed_results: list[sqlalchemy.Result[typing.Any]], exhaust: bool
) -> None:
    """The result holds a cursor, which stays open for as long as it is not closed."""
    dataset = stream.iter_sql(SHOPS_AND_AMOUNTS, executor)
    _ = next(dataset)

    assert not executed_results[0].closed

    if exhaust:
        _ = list(dataset)
    else:
        del dataset
        _ = gc.collect()

    assert executed_results[0].closed


def test_the_query_is_only_executed_once_iteration_starts(engine: sqlalchemy.Engine) -> None:
    """`iter_sql` is a generator, so a closed connection is only noticed on the first `next`."""
    conn = engine.connect()
    conn.close()

    dataset = stream.iter_sql(SHOPS_AND_AMOUNTS, conn)

    with pytest.raises(sqlalchemy.exc.ResourceClosedError):
        _ = next(dataset)


@pytest.fixture
def pokedb() -> sqlalchemy.Engine:
    engine = sqlalchemy.create_engine("sqlite://")  # in-memory

    # Load the fixtures
    here = os.path.dirname(os.path.realpath(__file__))
    with zipfile.ZipFile(os.path.join(here, "pokedb.zip")) as z:
        pokemons = json.loads(z.read("pokemons.json"))
        types = json.loads(z.read("types.json"))
        pokemon_types = json.loads(z.read("pokemon_types.json"))

    # Define the tables
    metadata = sqlalchemy.MetaData()

    t_pokemons = sqlalchemy.Table(
        "pokemons",
        metadata,
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("name", sqlalchemy.String),
        sqlalchemy.Column("HP", sqlalchemy.Integer),
        sqlalchemy.Column("Attack", sqlalchemy.Integer),
        sqlalchemy.Column("Defense", sqlalchemy.Integer),
        sqlalchemy.Column("Sp. Attack", sqlalchemy.Integer),
        sqlalchemy.Column("Sp. Defense", sqlalchemy.Integer),
        sqlalchemy.Column("Speed", sqlalchemy.Integer),
    )

    t_types = sqlalchemy.Table(
        "types",
        metadata,
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("name", sqlalchemy.String),
    )

    t_pokemon_types = sqlalchemy.Table(
        "pokemon_types",
        metadata,
        sqlalchemy.Column("pokemon_id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("type_id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("no", sqlalchemy.Integer, primary_key=True),
    )

    # Create the tables
    metadata.create_all(engine)

    # Insert the fixtures
    with engine.connect() as conn:
        _ = conn.execute(t_pokemons.insert(), pokemons)
        _ = conn.execute(t_types.insert(), types)
        _ = conn.execute(t_pokemon_types.insert(), pokemon_types)
        conn.commit()

    return engine


def test_iter_sql(pokedb: sqlalchemy.Engine) -> None:
    with pokedb.connect() as conn:
        dataset = stream.iter_sql(query="SELECT * FROM pokemons;", conn=conn)
        x, y = next(dataset)
        assert x["name"] == "Bulbasaur"
        assert y is None

    # This raises an exception because the resource is closed...
    with pytest.raises(sqlalchemy.exc.ResourceClosedError):
        for x, y in stream.iter_sql(query="SELECT * FROM pokemons;", conn=conn):
            pass

    # ... and yet we can still stream over the results because SQLAlchemy prefetches them
    x, y = next(dataset)
    assert x["name"] == "Ivysaur"

    # The Pokedex from generation 1 contains 151 pokemons, and we've already seen 2 of them
    assert sum(1 for _ in dataset) == 149

    # Check that the stream is depleted
    assert sum(1 for _ in dataset) == 0


def test_iter_sql_join(pokedb: sqlalchemy.Engine) -> None:
    query = """
        SELECT
            p.name,
            t1.name AS type_1,
            t2.name AS type_2
        FROM
            pokemons p,
            pokemon_types pt1,
            pokemon_types pt2,
            types t1,
            types t2
        WHERE
            pt1.no = 1 AND
            pt1.pokemon_id = p.id AND
            pt1.type_id = t1.id AND

            pt2.no = 2 AND
            pt2.pokemon_id = p.id AND
            pt2.type_id = t2.id;
    """

    with pokedb.connect() as conn:
        dataset = stream.iter_sql(query=query, conn=conn)
        x, _ = next(dataset)
        assert x["name"] == "Bulbasaur"
        assert x["type_1"] == "Grass"
        assert x["type_2"] == "Poison"
