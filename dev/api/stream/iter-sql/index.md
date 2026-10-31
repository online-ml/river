# iter\_sql

Iterates over the results from an SQL query.

By default, SQLAlchemy prefetches results. Therefore, even though you can iterate over the resulting rows one by one, the results are in fact loaded in batch. You can modify this behavior by configuring the connection you pass to `iter_sql`. For instance, you can set the `stream_results` parameter to `True`, as [explained in SQLAlchemy's documentation](<https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Connection.execution_options>). Note, however, that this isn't available for all database engines.

## Parameters

- **query**

  *Type* → `str | sqlalchemy.Executable`

  SQL query to be executed. Either a string, or any SQLAlchemy construct that can be executed, such as the result of `sqlalchemy.select`.
- **conn**

  *Type* → `sqlalchemy.Connection | Session`

  An SQLAlchemy construct which has an `execute` method. In other words you can pass a connection or a session.
- **target\_name**

  *Type* → `str | None`

  *Default* → `None`

  The name of the target field. If this is `None`, then `y` will also be `None`.

## Examples

As an example we'll create an in-memory database with SQLAlchemy.

```python
import datetime as dt
import sqlalchemy

engine = sqlalchemy.create_engine('sqlite://')

metadata = sqlalchemy.MetaData()

t_sales = sqlalchemy.Table('sales', metadata,
    sqlalchemy.Column('shop', sqlalchemy.String, primary_key=True),
    sqlalchemy.Column('date', sqlalchemy.Date, primary_key=True),
    sqlalchemy.Column('amount', sqlalchemy.Integer)
)

metadata.create_all(engine)

sales = [
    {'shop': 'Hema', 'date': dt.date(2016, 8, 2), 'amount': 20},
    {'shop': 'Ikea', 'date': dt.date(2016, 8, 2), 'amount': 18},
    {'shop': 'Hema', 'date': dt.date(2016, 8, 3), 'amount': 22},
    {'shop': 'Ikea', 'date': dt.date(2016, 8, 3), 'amount': 14},
    {'shop': 'Hema', 'date': dt.date(2016, 8, 4), 'amount': 12},
    {'shop': 'Ikea', 'date': dt.date(2016, 8, 4), 'amount': 16}
]

with engine.connect() as conn:
    _ = conn.execute(t_sales.insert(), sales)
    conn.commit()
```

We can now query the database. We will set `amount` to be the target field.

```python
from river import stream

with engine.connect() as conn:
    query = sqlalchemy.sql.select(t_sales)
    dataset = stream.iter_sql(query, conn, target_name='amount')
    for x, y in dataset:
        print(x, y)
```

```text
{'shop': 'Hema', 'date': datetime.date(2016, 8, 2)} 20
{'shop': 'Ikea', 'date': datetime.date(2016, 8, 2)} 18
{'shop': 'Hema', 'date': datetime.date(2016, 8, 3)} 22
{'shop': 'Ikea', 'date': datetime.date(2016, 8, 3)} 14
{'shop': 'Hema', 'date': datetime.date(2016, 8, 4)} 12
{'shop': 'Ikea', 'date': datetime.date(2016, 8, 4)} 16
```

This also with raw SQL queries.

```python
with engine.connect() as conn:
    query = "SELECT * FROM sales WHERE shop = 'Hema'"
    dataset = stream.iter_sql(query, conn, target_name='amount')
    for x, y in dataset:
        print(x, y)
```

```text
{'shop': 'Hema', 'date': '2016-08-02'} 20
{'shop': 'Hema', 'date': '2016-08-03'} 22
{'shop': 'Hema', 'date': '2016-08-04'} 12
```
