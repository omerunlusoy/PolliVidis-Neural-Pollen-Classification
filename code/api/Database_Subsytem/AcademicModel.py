# AcademicModel class is the corresponding class of Academic Table in the SQL database
class AcademicModel:

    # photo is PIL Image
    def __init__(self, academic_id, name, surname, appellation, institution, job_title, email, password, research_gate_link):
        self.academic_id = academic_id
        self.name = name
        self.surname = surname
        self.appellation = appellation
        self.institution = institution
        self.job_title = job_title
        self.email = email
        self.password = password
        self.research_gate_link = research_gate_link
