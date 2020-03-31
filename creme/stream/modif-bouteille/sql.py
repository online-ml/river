from sqlalchemy import create_engine, MetaData, select
import os

def iter_sql(con):
    """Yields rows from all tables in a database.
    
    Parameters:
        con: SQLAlchemy connectable (engine/connection) or database str URI.

    Yields:
        x: dict of features.
    """
    if isinstance(con, str):
        con = create_engine(con)

    meta = MetaData()
    meta.reflect(bind=con)
    count = 0 
    for table in meta.tables.keys():
        statement = select([meta.tables[table]])
        X = con.execute(statement).fetchall()
        feature_names = meta.tables[table].columns.keys()

        for row in X:
            yield dict(zip(feature_names, row))
