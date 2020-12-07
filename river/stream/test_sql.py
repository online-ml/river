import json
import os

import pytest
import sqlalchemy as sql
import zipfile

from river import stream


@pytest.fixture
def pokedb():
    engine = sql.create_engine("sqlite://")  # in-memory

    # Load the fixtures
    here = os.path.dirname(os.path.realpath(__file__))
    with zipfile.ZipFile(os.path.join(here, "pokedb.zip")) as z:
        pokemons = json.loads(z.read("pokemons.json"))
        types = json.loads(z.read("types.json"))
        pokemon_types = json.loads(z.read("pokemon_types.json"))

    # Define the tables
    metadata = sql.MetaData()

    t_pokemons = sql.Table(
        "pokemons",
        metadata,
        sql.Column("id", sql.Integer, primary_key=True),
        sql.Column("name", sql.String),
        sql.Column("HP", sql.Integer),
        sql.Column("Attack", sql.Integer),
        sql.Column("Defense", sql.Integer),
        sql.Column("Sp. Attack", sql.Integer),
        sql.Column("Sp. Defense", sql.Integer),
        sql.Column("Speed", sql.Integer),
    )

    t_types = sql.Table(
        "types",
        metadata,
        sql.Column("id", sql.Integer, primary_key=True),
        sql.Column("name", sql.String),
    )

    t_pokemon_types = sql.Table(
        "pokemon_types",
        metadata,
        sql.Column("pokemon_id", sql.Integer, primary_key=True),
        sql.Column("type_id", sql.Integer, primary_key=True),
        sql.Column("no", sql.Integer, primary_key=True),
    )

    # Create the tables
    metadata.create_all(engine)

    # Insert the fixtures
    with engine.connect() as conn:
        conn.execute(t_pokemons.insert(), pokemons)
        conn.execute(t_types.insert(), types)
        conn.execute(t_pokemon_types.insert(), pokemon_types)

    return engine


def test_iter_sql(pokedb):

    with pokedb.connect() as conn:
        dataset = stream.iter_sql(query="SELECT * FROM pokemons;", conn=conn)
        x, y = next(dataset)
        assert x["name"] == "Bulbasaur"
        assert y is None

    # This raises an exception because the resource is closed...
    with pytest.raises(sql.exc.StatementError):
        for x, y in stream.iter_sql(query="SELECT * FROM pokemons;", conn=conn):
            pass

    # ... and yet we can still stream over the results because SQLAlchemy prefetches them
    x, y = next(dataset)
    assert x["name"] == "Ivysaur"

    # The Pokedex from generation 1 contains 151 pokemons, and we've already seen 2 of them
    assert sum(1 for _ in dataset) == 149

    # Check that the stream is depleted
    assert sum(1 for _ in dataset) == 0


def test_iter_sql_join(pokedb):

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
        x, y = next(dataset)
        assert x["name"] == "Bulbasaur"
        assert x["type_1"] == "Grass"
        assert x["type_2"] == "Poison"
