from django.db import models

# Create your models here.
#class AcademicModel:
    
    # photo is PIL Image
   # def __init__(self, academic_id, name, surname, appellation, institution, job_title, email, password, photo, research_gate_link):
    #    self.academic_id = academic_id
    #    self.name = name
     #   self.surname = surname
      #  self.appellation = appellation
       # self.institution = institution
        #self.job_title = job_title
        #self.email = email
       # self.password = password
       # self.photo = photo
       # self.research_gate_link = research_gate_link


class Academic(models.Model):
     academic_id = models.IntegerField()
     name = models.CharField(max_length=200)
     surname = models.CharField(max_length=200)
     appellation = models.CharField(max_length=200)
     institution = models.CharField(max_length=200)
     job_title = models.CharField(max_length=200)
     email = models.CharField(max_length=200)
     password = models.CharField(max_length=200)
     photo = models.CharField(max_length=200)
     research_gate_link = models.CharField(max_length=200)

class Sample(models.Model):
    #id = models.IntegerField()
    sample_id = models.CharField(max_length=1000, unique= True)
    academic_id = models.IntegerField()
    sample_photo = models.ImageField(upload_to= 'uploads/') ## ???
    date = models.CharField(max_length=1000)
    location_latitude = models.FloatField()
    location_longitude = models.FloatField()
    analysis_text = models.CharField(max_length=1000)
    publication_status = models.BooleanField(default=False,blank=True, null=True)
    anonymous_status = models.BooleanField(default=False,blank=True, null=True)
    pollens = models.CharField(max_length=1000) ## ?????
    
