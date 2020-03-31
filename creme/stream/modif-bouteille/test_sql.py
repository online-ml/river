from sql import *
from sqlalchemy import create_engine, MetaData, select
import os

db_uri = 'sqlite:///veekun-pokedex.sqlite'
engine = create_engine(db_uri)
count = 1 

print("Test 1: Input URI",end="\n\n")
for x in iter_sql(db_uri):
    print(x) 
    if count == 5:
        count = 0
        break
    count += 1

print("\nTest 2: Input engine", end="\n\n")
for x in iter_sql(engine):
    print(x) 
    if count == 5:
        count = 0
        break
    count +=1 

print("\nTest 3: Input connected engine", end="\n\n")
for x in iter_sql(engine.connect()):
    print(x) 
    if count == 5:
        count = 0
        break
    count +=1 
