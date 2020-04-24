from sqlalchemy import create_engine, MetaData, Table, Column, Integer, VARCHAR

def init_database():
    db_uri = 'sqlite:///db.sqlite'
    engine = create_engine(db_uri)

# create table
    meta = MetaData(engine)
    table1 = Table('census', meta,
       Column('state', VARCHAR(30)),
       Column('sex', VARCHAR(1)),
       Column('age', Integer),
       Column('pop2000', Integer), 
       Column('pop2008', Integer)) 

    table2 = Table('state_fact', meta,
       Column('id', VARCHAR(256)),
       Column('name', VARCHAR(256)),
       Column('abbrevation', VARCHAR(256)),
       Column('country', VARCHAR(256)), 
       Column('type', VARCHAR(256)),
       Column('sort', VARCHAR(256)),
       Column('status', VARCHAR(256)),
       Column('occupied', VARCHAR(256)),
       Column('notes', VARCHAR(256)), 
       Column('fips_state', VARCHAR(256)), 
       Column('assoc_press', VARCHAR(256)),
       Column('standard_federal_region', VARCHAR(256)),
       Column('census_region', VARCHAR(256)), 
       Column('census_region_name', VARCHAR(256)),
       Column('census_division', VARCHAR(256)),
       Column('census_division_name', VARCHAR(256)), 
       Column('circuit_court', VARCHAR(256))
    )

    meta.create_all()

    conn = engine.connect()

# insert multiple data
    conn.execute(table1.insert(),[
    {'state': 'Illinois', 'sex': 'M', 'age': 0, 'pop2000': 89600, 'pop2008': 5012},
    {'state': 'Illinois', 'sex': 'M', 'age': 1, 'pop2000': 88445, 'pop2008': 91829},
    {'state': 'Illinois', 'sex': 'M', 'age': 2, 'pop2000': 88729, 'pop2008': 89547},
    {'state': 'Illinois', 'sex': 'M', 'age': 3, 'pop2000': 88868, 'pop2008': 90037},
    {'state': 'Illinois', 'sex': 'M', 'age': 4, 'pop2000': 91947, 'pop2008': 91111},
    {'state': 'Illinois', 'sex': 'M', 'age': 5, 'pop2000': 93894, 'pop2008': 89802},
    {'state': 'Illinois', 'sex': 'M', 'age': 6, 'pop2000': 93676, 'pop2008': 88931},
    {'state': 'Illinois', 'sex': 'M', 'age': 7, 'pop2000': 94818, 'pop2008': 90940},
    {'state': 'Illinois', 'sex': 'M', 'age': 8, 'pop2000': 95035, 'pop2008': 86943},
    {'state': 'Illinois', 'sex': 'M', 'age': 9, 'pop2000': 96436, 'pop2008': 86055},
    {'state': 'Illinois', 'sex': 'M', 'age': 10, 'pop2000': 97280, 'pop2008': 86565},
    {'state': 'Illinois', 'sex': 'M', 'age': 11, 'pop2000': 94029, 'pop2008': 86606},
    {'state': 'Illinois', 'sex': 'M', 'age': 12, 'pop2000': 92402, 'pop2008': 89596},
    {'state': 'Illinois', 'sex': 'M', 'age': 13, 'pop2000': 89926, 'pop2008': 91661},
    {'state': 'Illinois', 'sex': 'M', 'age': 14, 'pop2000': 90717, 'pop2008': 91256}
    ])

    conn.execute(table2.insert(), [
    {'id': '13', 'name': 'Illinois', 'abbreviation': 'IL', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '17', 'assoc_press': 'Ill.', 'standard_federal_region': 'V', 'census_region': '2', 'census_region_name': 'Midwest', 'census_division': '3', 'census_division_name': 'East North Central', 'circuit_court': '7'},
    {'id': '30', 'name': 'New Jersey', 'abbreviation': 'NJ', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '34', 'assoc_press': 'N.J.', 'standard_federal_region': 'II', 'census_region': '1', 'census_region_name': 'Northeast', 'census_division': '2', 'census_division_name': 'Mid-Atlantic', 'circuit_court': '3'},
    {'id': '34', 'name': 'North Dakota', 'abbreviation': 'ND', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '38', 'assoc_press': 'N.D.', 'standard_federal_region': 'VIII', 'census_region': '2', 'census_region_name': 'Midwest', 'census_division': '4', 'census_division_name': 'West North Central', 'circuit_court': '8'},
    {'id': '37', 'name': 'Oregon', 'abbreviation': 'OR', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '41', 'assoc_press': 'Ore.', 'standard_federal_region': 'X', 'census_region': '4', 'census_region_name': 'West', 'census_division': '9', 'census_division_name': 'Pacific', 'circuit_court': '9'},
    {'id': '51', 'name': 'Washington DC', 'abbreviation': 'DC', 'country': 'USA', 'type': 'capitol', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '11', 'assoc_press': '', 'standard_federal_region': 'III', 'census_region': '3', 'census_region_name': 'South', 'census_division': '5', 'census_division_name': 'South Atlantic', 'circuit_court': 'D.C.'},
    {'id': '49', 'name': 'Wisconsin', 'abbreviation': 'WI', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '55', 'assoc_press': 'Wis.', 'standard_federal_region': 'V', 'census_region': '2', 'census_region_name': 'Midwest', 'census_division': '3', 'census_division_name': 'East North Central', 'circuit_court': '7'},
    {'id': '3', 'name': 'Arizona', 'abbreviation': 'AZ', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '4', 'assoc_press': 'Ariz.', 'standard_federal_region': 'IX', 'census_region': '4', 'census_region_name': 'West', 'census_division': '8', 'census_division_name': 'Mountain', 'circuit_court': '9'},
    {'id': '4', 'name': 'Arkansas', 'abbreviation': 'AR', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '5', 'assoc_press': 'Ark.', 'standard_federal_region': 'VI', 'census_region': '3', 'census_region_name': 'South', 'census_division': '7', 'census_division_name': 'West South Central', 'circuit_court': '8'},
    {'id': '6', 'name': 'Colorado', 'abbreviation': 'CO', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '8', 'assoc_press': 'Colo.', 'standard_federal_region': 'VIII', 'census_region': '4', 'census_region_name': 'West', 'census_division': '8', 'census_division_name': 'Mountain', 'circuit_court': '10'},
    {'id': '11', 'name': 'Hawaii', 'abbreviation': 'HI', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '15', 'assoc_press': 'Hawaii', 'standard_federal_region': 'IX', 'census_region': '4', 'census_region_name': 'West', 'census_division': '9', 'census_division_name': 'Pacific', 'circuit_court': '9'},
    {'id': '16', 'name': 'Kansas', 'abbreviation': 'KS', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '20', 'assoc_press': 'Kan.', 'standard_federal_region': 'VII', 'census_region': '2', 'census_region_name': 'Midwest', 'census_division': '4', 'census_division_name': 'West North Central', 'circuit_court': '10'},
    {'id': '18', 'name': 'Louisiana', 'abbreviation': 'LA', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '22', 'assoc_press': 'La.', 'standard_federal_region': 'VI', 'census_region': '3', 'census_region_name': 'South', 'census_division': '7', 'census_division_name': 'West South Central', 'circuit_court': '5'},
    {'id': '26', 'name': 'Montana', 'abbreviation': 'MT', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '30', 'assoc_press': 'Mont.', 'standard_federal_region': 'VIII', 'census_region': '4', 'census_region_name': 'West', 'census_division': '8', 'census_division_name': 'Mountain', 'circuit_court': '9'},
    {'id': '27', 'name': 'Nebraska', 'abbreviation': 'NE', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '31', 'assoc_press': 'Nebr.', 'standard_federal_region': 'VII', 'census_region': '2', 'census_region_name': 'Midwest', 'census_division': '4', 'census_division_name': 'West North Central', 'circuit_court': '8'},
    {'id': '36', 'name': 'Oklahoma', 'abbreviation': 'OK', 'country': 'USA', 'type': 'state', 'sort': '10', 'status': 'current', 'occupied': 'occupied', 'notes': '', 'fips_state': '40', 'assoc_press': 'Okla.', 'standard_federal_region': 'VI', 'census_region': '3', 'census_region_name': 'South', 'census_division': '7', 'census_division_name': 'West South Central', 'circuit_court': '10'}
    ])


init_database()
