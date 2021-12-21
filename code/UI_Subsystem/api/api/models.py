from django.db import models

# Create your models here.
class AcademicModel:
    
    # photo is PIL Image
    def __init__(self, academic_id, name, surname, appellation, institution, job_title, email, password, photo, research_gate_link):
        self.academic_id = academic_id
        self.name = name
        self.surname = surname
        self.appellation = appellation
        self.institution = institution
        self.job_title = job_title
        self.email = email
        self.password = password
        self.photo = photo
        self.research_gate_link = research_gate_link
