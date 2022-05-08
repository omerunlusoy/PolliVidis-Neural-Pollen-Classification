#from asyncio.windows_events import NULL
import mysql.connector

import sys
#import AcademicModel,SampleModel,FeedbackModel,PollenTypeModel
#from FeedbackModel import FeedbackModelStatus

from .AcademicModel import AcademicModel
from .SampleModel import SampleModel
from .PollenTypeModel import PollenTypeModel
from .FeedbackModel import FeedbackModel
from .FeedbackModel import FeedbackModelStatus

from PIL import Image
import io
import datetime
import os
from random import randint


# Database Manager class implements all SQL queries

# Database Manager has the following services;
# connect_database()
# delete_tables()
# create_tables()
# initialize_pollen_types()

# get_academic_from_id(academic_id) -> AcademicModel
# get_academic_from_email(email) -> AcademicModel
# get_pollen_type(pollen_name) -> PollenTypeModel
# get_sample(sample_id) -> SampleModel
# get_samples_of_academic(academic_id) -> [SampleModel]
# get_samples_of_location(location_latitude, location_longitude) -> [SampleModel]
# get_all_samples() -> [SampleModel]
# get_total_sample_num() -> int
# get_feedback_from_feedback_id(feedback_id) -> FeedbackModel
# get_feedback_from_email(email) -> [FeedbackModel]

# IMPORTANT NOTE: do not have to fill primary key ids of parameter model for add functions.
# add_academic(AcademicModel) -> int
# add_sample(SampleModel) -> sample_id
# add_pollen_type(PollenTypeModel) -> Boolean
# add_feedback(FeedbackModel) -> Boolean

# delete_academic(academic_id) -> Boolean
# delete_academic_from_email(email) -> Boolean
# delete_sample(sample_id) -> Boolean
# delete_pollen_type(pollen_name) -> Boolean
# delete_feedback(feedback_id) -> Boolean

# update_sample(sample)
# update_academic(academic) -> Boolean          # uses academic_id to update all other variables (always use after get_academics)
# update_pollen_type_description(pollen)

# print_academic_table()
# print_sample_table()
# print_pollen_type_table()
# print_sample_has_pollen_table()
# print_feedback_table()

class Database_Manager:

    def __init__(self, initialize_database=False):
        self.connect_database()
        if initialize_database:
            self.delete_tables()
            self.create_tables()
            self.initialize_pollen_types()

    def connect_database(self):
        # for the first run, create database named pollividis
        # update connect info accordingly
        self.db = mysql.connector.connect(
            host="dijkstra.ug.bcc.bilkent.edu.tr",
            user="omer.unlusoy",
            password="8cWZ7QRc",
            database="omer_unlusoy"
        )
        self.cursor = self.db.cursor()
        # cursor.execute("CREATE DATABASE pollividis")  # run once

    def delete_tables(self):
        self.cursor.execute("DROP TABLE IF EXISTS Feedback;")
        self.cursor.execute("DROP TABLE IF EXISTS Sample_has_Pollen;")
        self.cursor.execute("DROP TABLE IF EXISTS Pollen_Type;")
        self.cursor.execute("DROP TABLE IF EXISTS Sample;")
        self.cursor.execute("DROP TABLE IF EXISTS Academic;")

    def create_tables(self):
        # Pollen_Type which holds all pollen species names and their analysis texts
        self.cursor.execute("CREATE TABLE Pollen_Type(" +
                            "pollen_name VARCHAR(50)," +
                            "explanation_text VARCHAR(10000) NOT NULL," +
                            "PRIMARY KEY(pollen_name));")

        self.cursor.execute("CREATE TABLE Academic(" +
                            "academic_id INT AUTO_INCREMENT," +
                            "name VARCHAR(50) NOT NULL," +
                            "surname VARCHAR(100) NOT NULL," +
                            "appellation VARCHAR(100)," +
                            "institution VARCHAR(1000)," +
                            "job_title VARCHAR(1000)," +
                            "email VARCHAR(200) UNIQUE," +
                            "password VARCHAR(50) NOT NULL," +
                            "photo BLOB," +
                            "research_gate_link VARCHAR(1000)," +
                            "PRIMARY KEY(academic_id));")

        self.cursor.execute("CREATE TABLE Sample(" +
                            "sample_id VARCHAR(50)," +
                            "academic_id INT NOT NULL," +
                            "sample_photo BLOB," +  # should not be stored in database but in the file system
                            "date VARCHAR(100)," +
                            "location_latitude DOUBLE," +
                            "location_longitude DOUBLE," +
                            "analysis_text VARCHAR(1000)," +
                            "publication_status BOOL," +
                            "anonymous_status BOOL," +
                            "FOREIGN KEY (academic_id) REFERENCES Academic(academic_id) ON DELETE NO ACTION ON UPDATE CASCADE," +
                            "PRIMARY KEY(sample_id));")

        self.cursor.execute("CREATE TABLE Sample_has_Pollen(" +
                            "sample_id VARCHAR(50) NOT NULL," +
                            "pollen_name VARCHAR(50) NOT NULL," +
                            "count INT NOT NULL," +
                            "FOREIGN KEY (sample_id) REFERENCES Sample(sample_id) ON DELETE CASCADE ON UPDATE CASCADE," +
                            "FOREIGN KEY (pollen_name) REFERENCES Pollen_Type(pollen_name) ON DELETE CASCADE ON UPDATE CASCADE," +
                            "PRIMARY KEY(sample_id, pollen_name));")

        self.cursor.execute("CREATE TABLE Feedback(" +
                            "feedback_id INT AUTO_INCREMENT," +
                            "academic_id INT," +
                            "name VARCHAR(50)," +
                            "email VARCHAR(200)," +
                            "text VARCHAR(1000) NOT NULL," +
                            "date DATETIME," +
                            "status ENUM('pending', 'answered') NOT NULL DEFAULT 'pending'," +
                            "FOREIGN KEY (academic_id) REFERENCES Academic(academic_id) ON DELETE CASCADE ON UPDATE CASCADE," +
                            "PRIMARY KEY(feedback_id));")

    def initialize_pollen_types(self):
        pollen_type_dict = [
        "ambrosia_artemisiifolia","alnus_glutinosa","acer_negundo","betula_papyrifera","juglans_regia","artemisia_vulgaris","populus_nigra","phleum_phleoides","picea_abies","juniperus_communis","ulmus_minor","quercus_robur","carpinus_betulus","ligustrum_robustrum","rumex_stenophyllus","ailanthus_altissima","thymbra_spicata","rubia_peregrina","olea_europaea","cichorium_intybus","chenopodium_album","borago_officinalis","acacia_dealbata"]
        for pollen in pollen_type_dict:
            #cur_pollen = PollenTypeModel(pollen, pollen_type_dict[pollen])
            cur_pollen = PollenTypeModel(pollen, " ")
            self.add_pollen_type(cur_pollen)

    def get_academic_from_id(self, academic_id):
        sql = "SELECT * FROM Academic WHERE academic_id = %s"
        val = (academic_id,)
        self.cursor.execute(sql, val)
        results = self.cursor.fetchall()

        if len(results) == 1:
            img = Image.open(io.BytesIO(results[0][8]))
            cur_academic = AcademicModel(results[0][0], results[0][1], results[0][2], results[0][3], results[0][4], results[0][5], results[0][6], results[0][7], img, results[0][9])
            return cur_academic
        else:
            print("Duplicate academic_id")
            return None

    def get_academic_from_email(self, email):
        sql = "SELECT * FROM Academic WHERE email = %s"
        val = (email,)
        self.cursor.execute(sql, val)
        results = self.cursor.fetchall()

        if len(results) == 1:
            #img = Image.open(io.BytesIO(results[0][8]))
            img = None
            cur_academic = AcademicModel(results[0][0], results[0][1], results[0][2], results[0][3], results[0][4], results[0][5], results[0][6], results[0][7], img, results[0][9])
            return cur_academic
        else:
            print("Duplicate or None academic_id")
            return None

    def get_pollen_type(self, pollen_name):
        sql = "SELECT * FROM Pollen_Type WHERE pollen_name = %s"
        val = (pollen_name,)
        self.cursor.execute(sql, val)
        results = self.cursor.fetchall()

        if len(results) == 1:
            cur_pollen_type = PollenTypeModel(results[0][0], results[0][1])
            return cur_pollen_type
        else:
            print("Duplicate or None pollen_name")
            return None

    def get_sample(self, sample_id):
        print(sample_id)
        sql = "SELECT * FROM Sample WHERE sample_id = %s"
        val = (sample_id,)
        self.cursor.execute(sql, val)
        results = self.cursor.fetchall()

        print(sample_id)
        print(len(results))
        #self.print_sample_table()
        if len(results) == 1:

            pollens = {}
            # we have found the sample, now lets fetch its pollens from Sample_has_Pollen Table
            sql = "SELECT * FROM Sample_has_Pollen WHERE sample_id = %s"
            val = (sample_id,)
            self.cursor.execute(sql, val)
            pollen_results = self.cursor.fetchall()

            for pol in pollen_results:
                pollens[pol[1]] = pol[2]

            sample_photo = Image.open(io.BytesIO(results[0][2]))
            cur_sample = SampleModel(results[0][0], results[0][1], sample_photo, results[0][3], results[0][4], results[0][5], results[0][6], results[0][7], results[0][8], pollens)
            return cur_sample
        else:
            print("Duplicate or None sample_id")
            return None

    def get_samples_of_academic(self, academic_id):
        sql = "SELECT * FROM Sample WHERE academic_id = %s"
        val = (academic_id,)
        self.cursor.execute(sql, val)
        results = self.cursor.fetchall()

        if len(results) > 0:

            print("NEW ORLEANS")

            samples = []
            for i in range(len(results)):

                pollens = {}
                # we have found the sample, now lets fetch its pollens from Sample_has_Pollen Table
                sql = "SELECT * FROM Sample_has_Pollen WHERE sample_id = %s"
                val = (results[i][0],)
                self.cursor.execute(sql, val)
                pollen_results = self.cursor.fetchall()


                for pol in pollen_results:
                    pollens[pol[1]] = pol[2]


                sample_photo = Image.open(io.BytesIO(results[0][2]))
                cur_sample = SampleModel(results[i][0], results[i][1], sample_photo, results[i][3], results[i][4], results[i][5], results[i][6], results[i][7], results[i][8], pollens)
                samples.append(cur_sample)

            return samples

        else:
            return []

    def get_samples_of_location(self, location_latitude, location_longitude):
        sql = "SELECT * FROM Sample WHERE location_latitude = %s and location_longitude = %s"
        val = (location_latitude, location_longitude,)
        self.cursor.execute(sql, val)
        results = self.cursor.fetchall()

        if len(results) > 0:

            samples = []
            for i in range(len(results)):

                pollens = {}
                # we have found the sample, now lets fetch its pollens from Sample_has_Pollen Table
                sql = "SELECT * FROM Sample_has_Pollen WHERE sample_id = %s"
                val = (results[i][0],)
                self.cursor.execute(sql, val)
                pollen_results = self.cursor.fetchall()

                for pol in pollen_results:
                    pollens[pol[1]] = pol[2]

                sample_photo = Image.open(io.BytesIO(results[0][2]))
                cur_sample = SampleModel(results[0][0], results[0][1], sample_photo, results[0][3], results[0][4], results[0][5], results[0][6], results[0][7], results[0][8], pollens)
                samples.append(cur_sample)

            return samples

        else:
            return []

    def get_all_samples(self):
        sql = "SELECT * FROM Sample"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()

        if len(results) > 0:

            samples = []
            for i in range(len(results)):

                pollens = {}
                # we have found the sample, now lets fetch its pollens from Sample_has_Pollen Table
                sql = "SELECT * FROM Sample_has_Pollen WHERE sample_id = %s"
                val = (results[i][0],)
                self.cursor.execute(sql, val)
                pollen_results = self.cursor.fetchall()

                for pol in pollen_results:
                    pollens[pol[1]] = pol[2]

                sample_photo = Image.open(io.BytesIO(results[0][2]))
                cur_sample = SampleModel(results[i][0], results[i][1], sample_photo, results[i][3], results[i][4], results[i][5], results[i][6], results[i][7], results[i][8], pollens)
                samples.append(cur_sample)

            return samples

        else:
            return []

    def get_total_sample_num(self):
        sql = "SELECT * FROM Sample"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        return len(results)

    def get_feedback_from_feedback_id(self, feedback_id):
        sql = "SELECT * FROM Feedback WHERE feedback_id = %s"
        val = (feedback_id,)
        self.cursor.execute(sql, val)
        results = self.cursor.fetchall()

        if len(results) == 1:
            if results[0][6] == 'pending':
                status = FeedbackModelStatus.pending
            else:
                status = FeedbackModelStatus.answered
            cur_feedback = FeedbackModel(results[0][0], results[0][1], results[0][2], results[0][3], results[0][4], results[0][5], status)
            return cur_feedback
        else:
            print("Duplicate or None feedback_id")
            return None

    def get_feedback_from_email(self, email):
        sql = "SELECT * FROM Feedback WHERE email = %s"
        val = (email,)
        self.cursor.execute(sql, val)
        results = self.cursor.fetchall()

        if len(results) > 0:
            feedbacks = []
            for i in range(len(results)):
                if results[i][6] == 'pending':
                    status = FeedbackModelStatus.pending
                else:
                    status = FeedbackModelStatus.answered
                cur_feedback = FeedbackModel(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[0][5], status)
                feedbacks.append(cur_feedback)

            return feedbacks
        else:
            return []

    def add_feedback(self, feedback):

        sql = "INSERT INTO Feedback (academic_id, name, email, text, date, status) " \
              "VALUES (%s, %s, %s, %s, %s, %s)"

        if isinstance(feedback, FeedbackModel):

            if feedback.status is FeedbackModelStatus.pending:
                status = "pending"
            else:
                status = "answered"

            val = (feedback.academic_id, feedback.name, feedback.email, feedback.text, feedback.date, status)
            try:
                self.cursor.execute(sql, val)
                self.db.commit()
                return True
            except(mysql.connector.Error, mysql.connector.Warning) as e:
                print(e)
                return False

        else:
            print("Given object is not type FeedbackModel")
            return False

    def delete_feedback(self, feedback_id):
        sql = "DELETE FROM Feedback WHERE feedback_id = %s"
        val = (feedback_id,)
        try:
            self.cursor.execute(sql, val)
            self.db.commit()
            return True
        except(mysql.connector.Error, mysql.connector.Warning) as e:
            print(e)
            return False

    def add_academic(self, academic):

        sql = "INSERT INTO Academic (name, surname, appellation, institution, job_title, email, password, photo, research_gate_link) " \
              "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        print("heey")
        self.print_academic_table()
        print(type(academic))
        print("hwat")
        if isinstance(academic, AcademicModel):
            if academic.photo != None:
                binaryData = None
            else:
                # to use open, we needed to save the image
                academic.photo.save("buff.jpg")
                with open("buff.jpg", 'rb') as file:
                    binaryData = file.read()
                os.remove("buff.jpg")

            val = (academic.name, academic.surname, academic.appellation, academic.institution, academic.job_title, academic.email, academic.password, binaryData,
                   academic.research_gate_link)

            #val = (academic.name, academic.surname, academic.appellation, academic.institution, academic.job_title, academic.email, academic.password, binaryData, "aaa")
            #val = ("ece","ece","ece","ece","ece","ece@ece","ece",None,"ece")
            #val = (academic.name, academic.surname, academic.appellation, academic.institution, academic.job_title, academic.email, academic.password, academic.photo,
            #       academic.research_gate_link)       
            try:
                print("heyo")
                self.cursor.execute(sql, val)
                print("peki bura?")
                self.db.commit()

                return self.get_academic_from_email(academic.email).academic_id
            except(mysql.connector.Error, mysql.connector.Warning) as e:
                print(e)
                return -1

        else:
            print("Given object is not type AcademicModel")
            return -1

    def delete_academic(self, academic_id):
        sql = "DELETE FROM Academic WHERE academic_id = %s"
        val = (academic_id,)
        try:
            self.cursor.execute(sql, val)
            self.db.commit()
            return True
        except(mysql.connector.Error, mysql.connector.Warning) as e:
            print(e)
            return False

    def delete_academic_from_email(self, email):
        sql = "DELETE FROM Academic WHERE email = %s"
        val = (email,)
        try:
            self.cursor.execute(sql, val)
            self.db.commit()
            return True
        except(mysql.connector.Error, mysql.connector.Warning) as e:
            print(e)
            return False

    def add_sample(self, sample):

        sql = "INSERT INTO Sample (academic_id, sample_photo, date, location_latitude, location_longitude, analysis_text, publication_status, anonymous_status) " \
              "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

        if isinstance(sample, SampleModel):
            # to use open, we needed to save the image
            sample.sample_photo.save("buff.jpg")
            with open("buff.jpg", 'rb') as file:
                binaryData = file.read()
            os.remove("buff.jpg")

            val = (sample.academic_id, binaryData, sample.date, sample.location_latitude, sample.location_longitude,
                   sample.analysis_text, sample.publication_status, sample.anonymous_status)
            try:
                self.cursor.execute(sql, val)
                self.db.commit()
            except(mysql.connector.Error, mysql.connector.Warning) as e:
                print(e)
                return -1

            sql = "SELECT MAX(sample_id) FROM Sample"
            self.cursor.execute(sql)
            results = self.cursor.fetchall()

            sample_id = -1
            if len(results) == 1:
                sample_id = results[0][0]
            else:
                return -1

            # we have added the sample to the Sample Table
            # we need to add its pollens to Sample_has_Pollen Table

            for pollen_name in sample.pollens:

                sql = "INSERT INTO Sample_has_Pollen (sample_id, pollen_name, count) VALUES (%s, %s, %s)"
                val = (sample_id, pollen_name, sample.pollens[pollen_name])  # DO NOT USE sample.sample_id
                try:
                    self.cursor.execute(sql, val)
                    self.db.commit()
                except(mysql.connector.Error, mysql.connector.Warning) as e:
                    print("Sample itself is added but one of its pollen types could not be added.")
                    print(e)
                    return -1

            return sample_id

        else:
            print("Given object is not type SampleModel")
            return -1

    def add_pollen_has(self,sample_id,pollen_name,pollen_count):
        sql = "INSERT INTO Sample_has_Pollen (sample_id, pollen_name, count) VALUES (%s, %s, %s)"
        val = (sample_id, pollen_name, pollen_count)  # DO NOT USE sample.sample_id
        try:
            self.cursor.execute(sql, val)
            self.db.commit()
            return 1
        except(mysql.connector.Error, mysql.connector.Warning) as e:
            print("Sample itself is added but one of its pollen types could not be added.")
            print(e)
            return -1    

    def delete_sample(self, sample_id):
        sql = "DELETE FROM Sample WHERE sample_id = %s"
        val = (sample_id,)
        try:
            self.cursor.execute(sql, val)
            self.db.commit()
            return True
        except(mysql.connector.Error, mysql.connector.Warning) as e:
            print(e)
            return False

    def add_pollen_type(self, pollen_type):

        sql = "INSERT INTO Pollen_Type (pollen_name, explanation_text) VALUES (%s, %s)"

        if isinstance(pollen_type, PollenTypeModel):
            val = (pollen_type.pollen_name, pollen_type.explanation_text)
            try:
                self.cursor.execute(sql, val)
                self.db.commit()
                return True
            except(mysql.connector.Error, mysql.connector.Warning) as e:
                print(e)
                return False
        else:
            print("Given object is not type PollenTypeModel")
            return False

    def delete_pollen_type(self, pollen_name):
        sql = "DELETE FROM Pollen_Type WHERE pollen_name = %s"
        val = (pollen_name,)
        try:
            self.cursor.execute(sql, val)
            self.db.commit()
            return True
        except(mysql.connector.Error, mysql.connector.Warning) as e:
            print(e)
            return False

    def update_academic(self, academic):

        if isinstance(academic, AcademicModel):

            sql = "UPDATE Academic SET name = %s, surname = %s, appellation = %s, institution = %s, job_title = %s, email = %s, password = %s, photo = %s, research_gate_link = %s" \
                  "WHERE academic_id = %s"

            # to use open, we needed to save the image
            academic.photo.save("buff.jpg")
            with open("buff2.jpg", 'rb') as file:
                binaryData = file.read()
            os.remove("buff2.jpg")

            val = (academic.name, academic.surname, academic.appellation, academic.institution, academic.job_title, academic.email, academic.password, binaryData,
                   academic.research_gate_link, academic.academic_id)
            try:
                self.cursor.execute(sql, val)
                self.db.commit()
                return True
            except(mysql.connector.Error, mysql.connector.Warning) as e:
                print(e)
                return False

        else:
            print("Given object is not type AcademicModel")
            return False
    
    def update_sample(self, sample):
    
        if isinstance(sample, SampleModel):

            #sql = "UPDATE Sample SET sample_id = %s, academic_id = %s,  date = %s, location_latitude = %s, location_longitude = %s, analysis_text = %s, publication_status = %s, anonymous_status = %s" \
            #      "WHERE sample_id = %s"
            sql = "UPDATE Sample SET analysis_text = %s WHERE sample_id = %s"
            # to use open, we needed to save the image
            '''
            sample.sample_photo.save("buff.jpg")
            with open("buff.jpg", 'rb') as file:
                binaryData = file.read()
            os.remove("buff.jpg")
            '''
            
            #val = (sample.sample_id, sample.academic_id,  sample.date, sample.location_latitude, sample.location_longitude, sample.analysis_text, sample.publication_status,
            #       sample.anonymous_status, sample.sample_id)
            print("in database analysis:")
            print(sample.analysis_text)
            print(sample.sample_id)
            val = (sample.analysis_text,sample.sample_id)
            try:
                
                self.cursor.execute(sql, val)
                self.db.commit()
                print("update complete!!!!!!")
                #return True
            except(mysql.connector.Error, mysql.connector.Warning) as e:
                print(e)
                return False
            


        else:
            print("Given object is not type SampleModel")
            return False
    

    def update_pollen_type_description(self, pollen):

        if isinstance(pollen, PollenTypeModel):

            sql = "UPDATE Pollen_Type SET explanation_text = %s WHERE pollen_name = %s"
            val = (pollen.explanation_text, pollen.pollen_name)
            try:
                self.cursor.execute(sql, val)
                self.db.commit()
                return True
            except(mysql.connector.Error, mysql.connector.Warning) as e:
                print(e)
                return False

        else:
            print("Given object is not type PollenTypeModel")
            return False

    def print_academic_table(self):
        sql = "SELECT * FROM Academic"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()

        if len(results) > 0:
            print("Academic Table:")
            for i in range(len(results)):
                print(i, ":", results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5], results[i][6], results[i][7], results[i][9])
            print("")
        else:
            print("No Academic Record...")

    def print_sample_table(self):
        sql = "SELECT * FROM Sample"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()

        if len(results) > 0:
            print("Sample Table:")
            for i in range(len(results)):
                print(i, ":", results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5], results[i][6], results[i][7], results[i][8])
            print("")
        else:
            print("No Sample Record...")

    def print_pollen_type_table(self):
        sql = "SELECT * FROM Pollen_Type"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()

        if len(results) > 0:
            print("Pollen_Type Table:")
            for i in range(len(results)):
                print(i, ":", results[i][0], results[i][1])
            print("")
        else:
            print("No Pollen_Type Record...")

    def print_sample_has_pollen_table(self):
        sql = "SELECT * FROM Sample_has_Pollen"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()

        if len(results) > 0:
            print("Sample_has_Pollen Table:")
            for i in range(len(results)):
                print(i, ":", results[i][0], results[i][1], results[i][2])
            print("")
        else:
            print("No Sample_has_Pollen Record...")

    def print_feedback_table(self):
        sql = "SELECT * FROM Feedback"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()

        if len(results) > 0:
            print("Feedback Table:")
            for i in range(len(results)):
                if results[0][6] == 'pending':
                    status = FeedbackModelStatus.pending
                else:
                    status = FeedbackModelStatus.answered
                print(i, ":", results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5], status)
            print("")
        else:
            print("No Feedback Record...")
