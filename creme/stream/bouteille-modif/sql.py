def iter_sql(query, con):
    """Yields over a query result.
    
    Parameters:
        query (str): SQL query to be executed.
        con (qlalchemy.engine.Engine): SQLAlchemy connectable (engine/connection).

    Yields:
        dict: dict of features.

    Example:
        >>> from creme import stream
        >>> engine = create_engine('your_db_uri')
        >>> metadata = MetaData().reflect(bind=engine)
        >>> with engine.connect() as con:
        ...     for row in stream.iter_sql(query="SELECT * FROM your_table_name", con=engine):
        ...         print(row)
    """ 
    
    result = con.execute(query)
    
    for row in result.fetchall():
        yield dict(zip(result.keys(), row))
