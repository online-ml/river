import typing


def iter_sql(query: typing.Union[str, 'sqlalchemy.sql.expression.Selectable'],
             conn: 'sqlalchemy.engine.Connectable', target_name: str = None):
    """Yields over a query result.

    Parameters:
        query (str or sqlalchemy.sql.expression.Selectable): SQL query to be executed.
        conn (sqlalchemy.engine.Connectable): An SQLAlchemy construct which has an ``execute``
            method. In other words you can pass an engine, a connection, or a session.
        target_name (str): The name of the target field. If this is ``None``, then ``y`` will also
            be ``None``.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    Example:

        ::

            As an example we'll create an in-memory database with SQLAlchemy.

            >>> import datetime as dt
            >>> import sqlalchemy

            >>> engine = sqlalchemy.create_engine('sqlite://')

            >>> metadata = sqlalchemy.MetaData()

            >>> t_sales = sqlalchemy.Table('sales', metadata,
            ...     sqlalchemy.Column('shop', sqlalchemy.String, primary_key=True),
            ...     sqlalchemy.Column('date', sqlalchemy.Date, primary_key=True),
            ...     sqlalchemy.Column('amount', sqlalchemy.Integer)
            ... )

            >>> metadata.create_all(engine)

            >>> sales = [
            ...     {'shop': 'Hema', 'date': dt.date(2016, 8, 2), 'amount': 20},
            ...     {'shop': 'Ikea', 'date': dt.date(2016, 8, 2), 'amount': 18},
            ...     {'shop': 'Hema', 'date': dt.date(2016, 8, 3), 'amount': 22},
            ...     {'shop': 'Ikea', 'date': dt.date(2016, 8, 3), 'amount': 14},
            ...     {'shop': 'Hema', 'date': dt.date(2016, 8, 4), 'amount': 12},
            ...     {'shop': 'Ikea', 'date': dt.date(2016, 8, 4), 'amount': 16}
            ... ]

            >>> with engine.connect() as conn:
            ...     _ = conn.execute(t_sales.insert(), sales)

            We can now query the database. We will set ``amount`` to be the target field.

            >>> from creme import stream

            >>> with engine.connect() as conn:
            ...     query = 'SELECT * FROM sales;'
            ...     X_y = stream.iter_sql(query, conn, target_name='amount')
            ...     for x, y in X_y:
            ...         print(x, y)
            {'shop': 'Hema', 'date': '2016-08-02'} 20
            {'shop': 'Ikea', 'date': '2016-08-02'} 18
            {'shop': 'Hema', 'date': '2016-08-03'} 22
            {'shop': 'Ikea', 'date': '2016-08-03'} 14
            {'shop': 'Hema', 'date': '2016-08-04'} 12
            {'shop': 'Ikea', 'date': '2016-08-04'} 16

    """

    result_proxy = conn.execute(query)

    if target_name is None:
        for row in result_proxy:
            yield dict(row.items()), None
        return

    for row in result_proxy:
        x = dict(row.items())
        y = x.pop(target_name)
        yield x, y
