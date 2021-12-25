from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.serializers import ModelSerializer
from .models import Academic, Sample
from ..managers.SignUpManager import sign_up
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import generics, status

from .serializers import SampleSerializer

import sys
sys.path.append("/Users/omerunlusoy/Desktop/CS 491/CS491_Senior_Design_Project/code")
print('anan', sys.path)
from Database_Subsytem.SampleModel import SampleModel
from Database_Subsytem.Database_Manager import Database_Manager
from ML_Subsystem.ML_Manager import ML_Manager
import json

from django.shortcuts import render
from django.http import JsonResponse

db_manager = Database_Manager(False)
ml_manager = ML_Manager()
# Create your views here.
@api_view(['POST'])
def analyses_post(request):
    
    #Database_Manager.connect_database()
    serializer = SampleSerializer(data=request.data)
    image, pollenText,pollens = ml_manager.analyze_sample(serializer.data['sample_photo'], "", serializer.data['date'], "John", db_manager)
    
    #serializer = SampleSerializer(data=request.data)
    #sampleObj = SampleModel(-1,-1, serializer.data['sample_photo'],serializer.data['location_latitude'],serializer.data['location_longitude'],serializer.data['analysis_text'],serializer.data['publication_status']serializer.data['anonymous_status'],serializer.data['pollens'])
    sampleObj = SampleModel(-1, 1, image,serializer.data['date'], serializer.data['location_latitude'], serializer.data['location_longitude'], pollenText, False, False, pollens)
    result = db_manager.add_sample(sampleObj)
    if( result== -1):
         return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def analyses_get_by_id(request,pk):
    #Database_Manager.connect_database()
    result = db_manager.get_sample(pk)

    if(result == None):
            return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result, status=status.HTTP_302_FOUND)
    #analyses = Sample.objects.get(id=pk)

@api_view(['GET'])
def get_all_samples(request):
    #Database_Manager.connect_database()
    result = db_manager.get_all_samples()
    
    if (result == []):
        return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)

    return Response(result, status=status.HTTP_302_FOUND)


#def sign_up(request):
#    print(request)
#
#    return HttpResponse(sign_up)

#def profile(request):
#    print(request)

#    return HttpResponse("Profile")

#def login(request):
#    print(request)

#    return HttpResponse("Login info")

#def gmap(request):
#    print(request)
    
#    return HttpResponse("Map info")