import sqlite3

connection =sqlite3.connect("students.db")

## create the cursor object to insert,record,create table , retrieve

cursor=connection.cursor()

table_info="""
Create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),SECTION VARCHAR(25),MARKS INT);
"""

cursor.execute(table_info)

cursor.execute('''Insert Into STUDENT values('Krish' , 'Data Science','A',90)''')


print("Data Inserted Successfully")

data=cursor.execute('''Select * From STUDENT''')

for row in data:
    print(row)

connection.commit()
connection.close()