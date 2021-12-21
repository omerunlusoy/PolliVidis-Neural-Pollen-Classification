from django.shortcuts import render
from django.http import HttpResponse
from .models  import AcademicModel
from ..managers.SignUpManager import sign_up
import json
# Create your views here.
def analyses(request):
    print(request)

    # Normally, get from database_manager
    test = AcademicModel()

    # Convert to json 
    jsonStr = json.dumps(test.__dict__)

    return HttpResponse(jsonStr)  

def sign_up(request):
    print(request)

    return HttpResponse(sign_up)

def profile(request):
    print(request)

    return HttpResponse("Profile")

def login(request):
    print(request)

    return HttpResponse("Login info")

def gmap(request):
    print(request)
    
    return HttpResponse("Map info")