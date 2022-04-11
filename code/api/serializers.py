from rest_framework import serializers
from .models import Sample,Academic


#class AcademicSerializer(serializers.ModelSerializer):
#    	class Meta:
#    		model = AcademicModel
#			fields =('academic_id', 'name', 'surname','appellation' ,'institution','job_title' ,'email' ,'password','photo','research_gate_link' )

#class SampleSerializer(serializers.ModelSerializer):
#	class Meta:
#		model = SampleModel
#		fields ='__all__'

class AcademicSerializer(serializers.ModelSerializer):
     class Meta:
         model = Academic
         fields = ('academic_id', 'name', 'surname','appellation' ,'institution','job_title' ,'email' ,'password','photo','research_gate_link')



class SampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sample
        fields = ('sample_id','academic_id','sample_photo','date','location_latitude','location_longitude'
                  ,'analysis_text','publication_status','anonymous_status','pollens')



