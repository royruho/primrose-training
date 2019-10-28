import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None


def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def drop_table(conn, drop_table_sql):
    try:
        c = conn.cursor()
        c.execute(drop_table_sql)
    except Error as e:
        print(e)


def select_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    for row in rows:
        print(row)


def insert_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)
    #return cur.lastrowid


def delete_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)


def LoadData(conn):
    sql_Load_queirs= """Delete from Doctor;
            INSERT INTO Doctor VALUES('111111111','DC','Dror','Cohen','Phd',2007,'Tel-Aviv');
            INSERT INTO Doctor VALUES('222222222','RM','Shimrit','Dabush','MA',2005,'Petah-Tikwa');
            INSERT INTO Doctor VALUES('333333333','DH','Tania','Reznikov','Msc',2010,'Herzelia');
            INSERT INTO Doctor VALUES('444444444','DB','Ayala','BenZvi','Phd',2009,'Tel-Aviv');
            
            Delete from Doctor_Specialization;
            INSERT INTO Doctor_Specialization VALUES('111111111','Computer');
            INSERT INTO Doctor_Specialization VALUES('111111111','Systems Engineering');
            INSERT INTO Doctor_Specialization VALUES('222222222','Systems Engineering');
            INSERT INTO Doctor_Specialization VALUES('444444444','Artificial Intelligence');
            INSERT INTO Doctor_Specialization VALUES('444444444','DataBase');
            INSERT INTO Doctor_Specialization VALUES('111111111','Engineering');
            INSERT INTO Doctor_Specialization VALUES('333333333','Engineering');
            
            
            Delete from Patients;
            INSERT INTO Patients VALUES('777777777','PP','Dor','Matazfi','03/10/2001','bla','Rona','David');
            INSERT INTO Patients VALUES('888888888','CC','Dror','Samet','23/05/1991','blabla','Tiki','Dani');
            INSERT INTO Patients VALUES('787878787','SS','Galit','Naim','12/07/1997','blab','Ruth','Moshe');
            INSERT INTO Patients VALUES('999999999','EE','Guy','shani','15/11/1986','bbbb','Orna','Asher');
            INSERT INTO Patients VALUES('191919191','BB','Mordechai','BenDror','25/10/2000','bbbb','Tami','Oren');
            INSERT INTO Patients VALUES('232323232','EE','Shiran','Aliashev','08/12/1985','bbabb','Lea','Shlomi');
            
            Delete from Treatment;
            INSERT INTO Treatment VALUES(111,'Name1','Bla Bla Bla Bla');
            INSERT INTO Treatment VALUES(333,'Name2','Bla Bla');
            INSERT INTO Treatment VALUES(444,'Name3','Bla Bla Bla');
            INSERT INTO Treatment VALUES(555,'Name4','Bla');
            
            
            Delete from Shifts;
            INSERT INTO Shifts VALUES('111111111','01/02/2019','13:00','20:00');
            INSERT INTO Shifts VALUES('222222222','02/02/2019','13:00','20:00');
            INSERT INTO Shifts VALUES('444444444','01/02/2019','14:00','21:00');
            INSERT INTO Shifts VALUES('111111111','03/02/2019','09:00','16:00');
            INSERT INTO Shifts VALUES('333333333','01/02/2019','13:00','21:00');
            INSERT INTO Shifts VALUES('222222222','05/02/2019','13:00','20:00');
            
            Delete from Labs;
            INSERT INTO Labs VALUES(99,'NameLab1','bla');
            INSERT INTO Labs VALUES(88,'NameLab2','blabla');
            INSERT INTO Labs VALUES(11,'NameLab3','blablabal');
            INSERT INTO Labs VALUES(77,'NameLab4','bbbbbb');
            INSERT INTO Labs VALUES(33,'NameLab5','aaaa');
            INSERT INTO Labs VALUES(44,'NameLab6','blablabla');
            
            Delete from Patients_Labs;
            INSERT INTO Patients_Labs VALUES(99,'01/01/2019','777777777','Good');
            INSERT INTO Patients_Labs VALUES(11,'01/01/2019','777777777','VeryGood');
            INSERT INTO Patients_Labs VALUES(99,'12/12/2018','777777777','Bad');
            INSERT INTO Patients_Labs VALUES(44,'24/05/2018','999999999','Good');
            INSERT INTO Patients_Labs VALUES(33,'01/01/2019','999999999','VeryGood');
            INSERT INTO Patients_Labs VALUES(99,'01/01/2019','787878787','Very Very Good');
            INSERT INTO Patients_Labs VALUES(77,'12/12/2018','999999999','Good');
            INSERT INTO Patients_Labs VALUES(11,'02/02/2018','888888888','Bad');
            INSERT INTO Patients_Labs VALUES(33,'22/03/2019','191919191','Bad');
            
            
            Delete from Patinets_Treatments;
            INSERT INTO Patinets_Treatments VALUES('999999999',333,'333333333','07/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('888888888',555,'333333333','07/11/2018','12/07/2019');
            INSERT INTO Patinets_Treatments VALUES('999999999',111,'333333333','07/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('777777777',333,'111111111','17/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('777777777',111,'444444444','27/06/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('191919191',555,'444444444','01/01/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('191919191',111,'222222222','15/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('888888888',333,'222222222','01/01/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('999999999',555,'222222222','07/05/2019','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('777777777',444,'444444444','27/07/2018','15/07/2019');
            INSERT INTO Patinets_Treatments VALUES('777777777',555,'444444444','27/08/2018','15/07/2019');
            
            
            Delete from Patinets_Progress;
            INSERT INTO Patinets_Progress
            VALUES('999999999',333,'333333333','bla','bla bla','22/08/2018');
            INSERT INTO Patinets_Progress
            VALUES('999999999',111,'333333333','blabla','bla','30/07/2018');
            INSERT INTO Patinets_Progress
            VALUES('777777777',333,'111111111','bla','bla','15/10/2017');
            INSERT INTO Patinets_Progress
            VALUES('777777777',111,'444444444','blablabla','bla','25/12/2018');
            INSERT INTO Patinets_Progress
            VALUES('999999999',555,'222222222','bla','bla','15/12/2018');
            INSERT INTO Patinets_Progress 
            VALUES('777777777',444,'444444444','bla bla','bla','13/11/2018');
            INSERT INTO Patinets_Progress
            VALUES('777777777',555,'444444444','bla','bla','30/07/2018');"""
    queries = sql_Load_queirs.split(";")
    for query in queries:
        insert_query(conn, query)


def createAllTables(conn):

    #### Example- Create Doctor Table####
    sql_drop_table = """DROP TABLE IF EXISTS Doctor"""
    drop_table(conn,sql_drop_table)
    sql_create_doctors_table = """CREATE TABLE  IF NOT EXISTS Doctor
                                    (
                                       ID			char(9)	primary key,
                                       Psw          varchar(20) not null,
                                       First_Name	varchar(50) not null,
                                       Last_Name	varchar(50) not null,
                                       Degree       varchar(20),
                                       P_Year       char(10),
                                       Address      varchar(50)
                                    );
                                    """
    create_table(conn, sql_create_doctors_table)
    
    #### Create	Specialize Table ####
    sql_drop_table = """DROP TABLE IF EXISTS Doctor_Specialization"""
    drop_table(conn,sql_drop_table)
    sql_create_Specialize_table = """CREATE TABLE  IF NOT EXISTS Doctor_Specialization
                                    (
                                       ID			char(9)	,
                                       Specialty    varchar(50) ,
                                       PRIMARY KEY (ID ,Specialty),
                                       FOREIGN KEY (ID) REFERENCES Doctor(ID) 
                                    );
                                    """
    create_table(conn, sql_create_Specialize_table)
    
        #### Create	Shifts Table ####
    sql_drop_table = """DROP TABLE IF EXISTS Shifts"""
    drop_table(conn,sql_drop_table)
    sql_create_Shifts_table = """CREATE TABLE  IF NOT EXISTS Shifts
                                    (
                                       ID			char(9)	not null,
                                       Date    varchar(20)  not null,
                                       start_time    char(10)  not null,
                                       finish_time    char(10) not null
                                       CHECK (cast(finish_time as date) > cast(start_time as date)),
                                       FOREIGN KEY (ID) REFERENCES Doctor(ID), 
                                       PRIMARY KEY (ID, Date, start_time)
                                    );
                                    """
    create_table(conn, sql_create_Shifts_table)
    
            #### Create	Patients Table ####
    sql_create_Patients_table = """CREATE TABLE  IF NOT EXISTS Patients
                                    (
                                       Patient_ID	char(9)	PRIMARY KEY,
                                       Patient_Password varchar(20),
                                       First_Name varchar(20),
                                       Last_Name varchar(20),
                                       Birthday varchar(20) ,
                                       Address varchar(50),
                                       Fathers_name varchar(20),
                                       Mothers_name varchar(20)
                                    );
                                    """
    create_table(conn, sql_create_Patients_table)
    
                #### Create	Treatment Table ####

    
    sql_create_Treatment_table = """CREATE TABLE  IF NOT EXISTS Treatment
                                    (
                                    Code INT PRIMARY KEY,
                                    Treatment_Name varchar(50),
                                    Treatment_description varchar(50)
                                    );
                                    """
    create_table(conn, sql_create_Treatment_table)    
    
                #### Create	Labs Table ####
    sql_create_Labs_table = """CREATE TABLE  IF NOT EXISTS Labs
                                (
                                Lab_ID char(9) PRIMARY KEY,
                                Lab_name varchar(50),
                                Lab_description varchar(50)
                                );
                                """
    create_table(conn, sql_create_Labs_table)
    
                    #### Create	Labs Table ####
    sql_create_Patients_Labs_table = """CREATE TABLE  IF NOT EXISTS Patients_Labs
                                (
                                Lab_ID char(9),
                                Lab_Date varchar(20),
                                Patient_id char(9),
                                Result varchar(50),
                                PRIMARY KEY (Lab_ID , Lab_Date, Patient_id),
                                FOREIGN KEY (Lab_ID) REFERENCES Labs(Lab_ID),
                                FOREIGN KEY (Patient_id) REFERENCES Patients(Patients_ID)
                                );
                                """
    create_table(conn, sql_create_Patients_Labs_table)
    
    sql_create_Patinets_Treatments_table = """CREATE TABLE  IF NOT EXISTS Patinets_Treatments
                            (
                            Patient_ID char(9),
                            Treatment_code INT,
                            Doctor_id char(9),
                            Start_date varchar(50),
                            Finish_date varchar(50),
                            PRIMARY KEY (Patient_ID , Treatment_code, Doctor_id)
                            FOREIGN KEY (Doctor_id) REFERENCES Doctor(ID),
                            FOREIGN KEY (Patient_id) REFERENCES Patients(Patient_ID),
                            FOREIGN KEY (Treatment_code) REFERENCES Treatment(Code)
                            );
                            """
    create_table(conn, sql_create_Patinets_Treatments_table)
    
    sql_create_Patinets_Progress_table = """CREATE TABLE  IF NOT EXISTS patinets_Progress
                            (
                            Patient_ID char(9),
                            Treatment_code INT,
                            Doctor_id char(9),
                            Initial_description varchar(50),
                            Present_description varchar(50),
                            Date_Ending varchar(50),
                            PRIMARY KEY (Patient_ID , Treatment_code, Doctor_id),
                            FOREIGN KEY (Doctor_id) REFERENCES Doctor(ID),
                            FOREIGN KEY (Patient_id) REFERENCES Patients(Patient_ID),
                            FOREIGN KEY (Treatment_code) REFERENCES Treatment(Code)
                            );
                            """
    create_table(conn, sql_create_Patinets_Progress_table)    
    #Your code in here- Create all other tables


def createAllqueries(conn):
    #### Example- Q1####
    print ("qeury 1:")
    sql_Q1_query = """select First_Name,Last_Name
                                    from Doctor,shifts
                                    where Doctor.ID=shifts.id and shifts.Date='01/02/2019'
                                    ; """
    select_query(conn,sql_Q1_query)
    
    
    print ("\n qeury 2:")
    sql_Q2_query = """select First_Name,Last_Name
                                    from Doctor,shifts
                                    where Doctor.ID=shifts.id and shifts.Date='01/02/2019' 
                                    and shifts.start_time = '13:00'
                                    ; """
    select_query(conn,sql_Q2_query)
    print ("\n qeury 3:")
    sql_Q3_query = """select First_Name,Last_Name,Lab_name
                                    from patients P, labs L, Patinets_Labs PL 
                                    where L.lab_ID  = pl.lab_id 
                                    and P.Patient_ID=PL.Patient_id 
                                    and PL.lab_date = '01/01/2019'
                                    ; """
    select_query(conn,sql_Q3_query)
    
    print ("\n qeury 4:")
    sql_Q4_query = """select Specialty
                                    from Doctor_Specialization DS, Doctor D
                                    where DS.ID = D.ID and D.first_name = 'Ayala' and D.last_name ='BenZvi'
                                    ; """
    select_query(conn,sql_Q4_query)
    
    print ("\n qeury 5:")
    sql_Q5_query = """select D.ID ,first_name, Last_name from Doctor D
                                    WHERE D.ID in ( 
                                    select DS.ID from Doctor_Specialization DS
                                    group by DS.ID 
                                    order by COUNT(*) desc
                                    limit 1)
                                    ; """
    select_query(conn,sql_Q5_query)
    
    print ("\n qeury 6:")
    sql_Q6_query = """ select P.patient_Id , P.first_name, P.last_name from Patients P
                        where p.patient_id = (
                        select Patient_Id from Patients_Labs 
                        EXCEPT 
                        select Patient_Id from Patinets_Treatments)
                                    ; """
    select_query(conn,sql_Q6_query)

    print ("\n qeury 7:")
    sql_Q7_query = """ select ID, First_name from Doctor 
                        where doctor.ID in (
                       select PT.doctor_Id from Patinets_Treatments PT
                       group by PT.doctor_Id
                       order by COUNT(DISTINCT PT.Patient_Id) desc
                       limit  3)

                                    ; """
    select_query(conn,sql_Q7_query)
    print ("\n qeury 8:")
    sql_Q8_query = """ select patient_Id, First_name from Patients 
                       where patient_Id in (
                       select PT.patient_Id from Patinets_Treatments PT
                       group by PT.patient_Id
                       order by COUNT(DISTINCT PT.Treatment_code) desc
                       limit  2)

                                    ; """
    select_query(conn,sql_Q8_query)
    print ("\n qeury 9:")
    sql_Q9_query = """select p.patient_id , p.first_name from Patients as p where p.patient_id in 
                       ( select patient_id from (select patient_Id, COUNT(Treatment_code) as CTC from 
                       Patinets_Treatments group by Patinets_Treatments.patient_Id) where CTC > 3) 
                                    ; """
    select_query(conn,sql_Q9_query)
    
    print ("\n qeury 10:")
    sql_Q10_query = """select p.patient_id , p.first_name from Patients p where p.patient_id in 
                       (select p.patient_id from Patients as p 
                       EXCEPT  
                       select patient_id from (select patient_Id, COUNT(Treatment_code) as CTC 
                       from Patinets_Treatments group by Patinets_Treatments.patient_Id) where CTC > 0 )
                                    ; """
    select_query(conn,sql_Q10_query)
    print ("\n qeury 11:")
    sql_Q11_query = """select p.patient_id , p.first_name from Patients p where p.patient_id in
                       (select DISTINCT pt1.patient_id from Patinets_Treatments pt1 where  pt1.Treatment_code in 
                       (select pt.Treatment_code from Patinets_Treatments pt where pt.patient_Id = '999999999') 
                       and pt1.patient_id <> '999999999')
                                    ; """
    select_query(conn,sql_Q11_query)
    print ("\n qeury 12:")
    sql_Q12_query = """ select p.patient_id , p.first_name, p.last_name from Patients p where p.patient_id in
                        (select p.patient_id from Patients as p 
                        EXCEPT 
                        select DISTINCT pp.patient_id from Patinets_progress as pp 
                        where pp.present_description IS NOT NULL 
                        or  pp.Initial_description IS NOT NULL)                
                       ; """
    select_query(conn,sql_Q12_query)
    print ("\n qeury 13:")
    sql_Q13_query = """ select p.patient_id , p.first_name, p.last_name from Patients p where p.patient_id in
                        (select pp.patient_ID from Patinets_progress as pp
                        group by pp.patient_ID 
                        order by COUNT(*) desc
                        limit 1)
                    ; """
    select_query(conn,sql_Q13_query)
    
    print ("\n qeury 14: insert query")
    sql_Q14_query = """ INSERT INTO Patinets_Treatments VALUES('787878787',555,'444444444','27/08/2019','15/09/2019')
                    ; """
    insert_query(conn, sql_Q14_query)
    print ("\n qeury 14: ",sql_Q14_query, " complete")
    
    print ("\n qeury 15: insert query")
    sql_Q15_query = """ INSERT INTO Patinets_progress VALUES('787878787',555,'444444444','bla','bla','27/09/2019')
                    ; """
    insert_query(conn, sql_Q15_query)
    print ("\n qeury 15: ",sql_Q15_query, " complete")
    
    print ("\n qeury 16: deletion query")
    sql_Q16_query = """ DELETE FROM Patinets_Treatments 
                        WHERE patient_ID = '777777777'
                    ; """
    delete_query(conn, sql_Q16_query)
    print ("\n qeury 16: ",sql_Q16_query, " complete")
    print ("\n qeury 17: update query")
    sql_Q17_query = """ UPDATE Doctor 
                        SET degree = 'phd'
                        WHERE doctor.ID = '222222222' 
                    ; """
    delete_query(conn, sql_Q17_query)
    print ("\n qeury 17: ",sql_Q17_query, " complete")

#     Your code in here- Answer all other question+


def main():
    database = r"C:\Users\royru\Desktop\primrose\github\primrose-training\SQL\Hospital.db"
    conn = create_connection(database)
    if conn is not None:
        createAllTables(conn)
        LoadData(conn)
        createAllqueries(conn)
    else:
        print("Error! cannot create the database connection.")


if __name__ == '__main__':
    main()
