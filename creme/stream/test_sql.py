from . import sql
from . import create_database
from sqlalchemy import create_engine, MetaData 
import pytest

@pytest.mark.parametrize("query,expected",[
    ("SELECT * FROM census", {
        ('state', 'sex', 'age', 'pop2000', 'pop2008'): [
            ['Illinois', 'M', 0, 89600, 5012],
            ['Illinois', 'M', 1, 88445, 91829],
            ['Illinois', 'M', 2, 88729, 89547],
            ['Illinois', 'M', 3, 88868, 90037],
            ['Illinois', 'M', 4, 91947, 91111],
            ['Illinois', 'M', 5, 93894, 89802],
            ['Illinois', 'M', 6, 93676, 88931],
            ['Illinois', 'M', 7, 94818, 90940],
            ['Illinois', 'M', 8, 95035, 86943],
            ['Illinois', 'M', 9, 96436, 86055],
            ['Illinois', 'M', 10, 97280, 86565],
            ['Illinois', 'M', 11, 94029, 86606],
            ['Illinois', 'M', 12, 92402, 89596],
            ['Illinois', 'M', 13, 89926, 91661],
            ['Illinois', 'M', 14, 90717, 91256]
        ]}          
    ),
    ("SELECT * FROM state_fact", {
        ('id', 'name', 'abbrevation', 'country', 'type', 'sort', 
         'status', 'occupied', 'notes', 'fips_state', 'assoc_press',
         'standard_federal_region', 'census_region', 'census_region_name',
         'census_division', 'census_division_name', 'circuit_court'): [
            ['13', 'Illinois', None, 'USA', 'state', '10', 'current', 'occupied',
            '', '17', 'Ill.', 'V', '2', 'Midwest', '3', 'East North Central', '7'],
            ['30', 'New Jersey', None, 'USA', 'state', '10', 'current', 'occupied',
            '', '34', 'N.J.', 'II', '1', 'Northeast', '2', 'Mid-Atlantic', '3'],
            ['34', 'North Dakota', None, 'USA', 'state', '10', 'current', 'occupied',
            '', '38', 'N.D.', 'VIII', '2', 'Midwest', '4', 'West North Central', '8'],
            ['37', 'Oregon', None, 'USA', 'state', '10', 'current', 'occupied',
            '', '41', 'Ore.', 'X', '4', 'West', '9', 'Pacific', '9'],
            ['51', 'Washington DC', None, 'USA', 'capitol', '10', 'current',
            'occupied', '', '11', '', 'III', '3', 'South', '5', 'South Atlantic', 'D.C.'],
            ['49', 'Wisconsin', None, 'USA', 'state', '10', 'current', 'occupied',
            '', '55', 'Wis.', 'V', '2', 'Midwest', '3', 'East North Central', '7'],
            ['3', 'Arizona', None, 'USA', 'state', '10', 'current', 'occupied', '',
            '4', 'Ariz.', 'IX', '4', 'West', '8', 'Mountain', '9'],
            ['4', 'Arkansas', None, 'USA', 'state', '10', 'current', 'occupied',
            '', '5', 'Ark.', 'VI', '3', 'South', '7', 'West South Central', '8'],
            ['6', 'Colorado', None, 'USA', 'state', '10', 'current', 'occupied', 
            '', '8', 'Colo.', 'VIII', '4', 'West', '8', 'Mountain', '10'],
            ['11', 'Hawaii', None, 'USA', 'state', '10', 'current', 'occupied',
            '', '15', 'Hawaii', 'IX', '4', 'West', '9', 'Pacific', '9'],
            ['16', 'Kansas', None, 'USA', 'state', '10', 'current', 'occupied', 
            '', '20', 'Kan.', 'VII', '2', 'Midwest', '4', 'West North Central', '10'],
            ['18', 'Louisiana', None, 'USA', 'state', '10', 'current', 'occupied', 
            '', '22', 'La.', 'VI', '3', 'South', '7', 'West South Central', '5'],
            ['26', 'Montana', None, 'USA', 'state', '10', 'current', 'occupied', 
            '', '30', 'Mont.', 'VIII', '4', 'West', '8', 'Mountain', '9'],
            ['27', 'Nebraska', None, 'USA', 'state', '10', 'current', 'occupied', 
            '', '31', 'Nebr.', 'VII', '2', 'Midwest', '4', 'West North Central', '8'],
            ['36', 'Oklahoma', None, 'USA', 'state', '10', 'current', 'occupied', 
            '', '40', 'Okla.', 'VI', '3', 'South', '7', 'West South Central', '10']
        ]}
    )
])

def initialization():
    # Create the database file.
    create_database.init_database()

    db_uri = 'sqlite:///db.sqlite'
    engine = create_engine(db_uri)

    return engine 

def test_sql(query, expected):
    expected_keys = list(expected.keys())
    expected_values = list(expected.values())[0]
    engine = initialization()

    with engine.connect() as con:
        for index, row in enumerate(sql.iter_sql(query, con)):
            assert list(row.keys()) == expected_keys()
            assert list(row.keys()) == expected_values[index]

