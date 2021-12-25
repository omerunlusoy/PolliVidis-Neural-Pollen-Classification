import datetime
from PIL import Image
from Database_Manager import Database_Manager

from AcademicModel import AcademicModel
from SampleModel import SampleModel
from PollenTypeModel import PollenTypeModel
from FeedbackModel import FeedbackModel
from FeedbackModel import FeedbackModelStatus

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

# update_academic(academic) -> Boolean          # uses academic_id to update all other variables (always use after get_academics)
# update_pollen_type_description(pollen)

# print_academic_table()
# print_sample_table()
# print_pollen_type_table()
# print_sample_has_pollen_table()
# print_feedback_table()


# Database_Manager Tests
# at each run, creating Database_Manager object resets database
# for the first run, create database named pollividis
# check database connection info
database_manager = Database_Manager(initialize_database=True)

# add_academic test
pp = Image.open("6.jpg")
academic = AcademicModel(-1, "John", "SaysHi", "Dr.", "Bilkent University", "Researcher", "john@bilkent.edu.tr", "asd123", pp, "research_gate/John")
john_id = database_manager.add_academic(academic)
academic_id = database_manager.get_academic_from_email("john@bilkent.edu.tr").academic_id
print(john_id, academic_id)
academic = database_manager.get_academic_from_id(academic_id)
print(academic.name)
database_manager.print_academic_table()

# add_pollen_type test
pollen_type1 = PollenTypeModel("Betula", "betula description")
pollen_type2 = PollenTypeModel("Rumex", "rumex description")
database_manager.add_pollen_type(pollen_type1)
database_manager.add_pollen_type(pollen_type2)
database_manager.print_pollen_type_table()

# add_sample test
sample_image = Image.open("6.jpg")
pollens = {"Betula": 10, "Rumex": 16}
sample1 = SampleModel(-1, 1, sample_image, datetime.datetime.now(), 1.1, 1.1, "analysis", False, False, pollens)
sample_id1 = database_manager.add_sample(sample1)
pollens = database_manager.get_sample(sample_id1).pollens
print(pollens)
database_manager.print_sample_table()

# prints sample1
sample1 = database_manager.get_sample(sample_id1)
print(sample1.academic_id)
print(database_manager.get_total_sample_num())

# get_academic and get_sample test
academic_object1 = database_manager.get_academic_from_id(1)
academic_object2 = database_manager.get_academic_from_email('john@bilkent.edu.tr')
print(academic_object1.email, academic_object2.name)
print(sample1.sample_id, sample1.date, sample1.pollens, sample1.anonymous_status)

feedback1 = FeedbackModel(-1, 1, "John", "a@b", "1234", datetime.datetime.now(), FeedbackModelStatus.pending)
database_manager.add_feedback(feedback1)
database_manager.print_feedback_table()

samp2 = database_manager.get_samples_of_academic(academic_object1.academic_id)
print(samp2[0].sample_id)

# update
academic_object1.name = "John Updated"
database_manager.update_academic(academic_object1)
database_manager.print_academic_table()

pol1 = database_manager.get_pollen_type(pollen_type1.pollen_name)
pol1.explanation_text += " Updated"
database_manager.update_pollen_type_description(pol1)
database_manager.print_pollen_type_table()

# delete
feedback2 = database_manager.get_feedback_from_email("a@b")[0].feedback_id
database_manager.delete_feedback(feedback2)
database_manager.delete_sample(sample1.sample_id)
# database_manager.delete_academic(academic_object1.academic_id)
database_manager.delete_pollen_type("Betula")

database_manager.print_sample_table()
database_manager.print_academic_table()
database_manager.print_pollen_type_table()
database_manager.print_feedback_table()

