import django
from django.forms import model_to_dict
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.core import serializers
import json

from api.Database_Subsytem.AcademicModel import AcademicModel

from .models import Sample, Academic
from .serializers import AcademicSerializer, SampleSerializer,FeedbackSerializer

from .Database_Subsytem.SampleModel import SampleModel
from .Database_Subsytem.Database_Manager import Database_Manager
from .Database_Subsytem.FeedbackModel import FeedbackModel

#from .ML_Subsystem.ML_Manager import ML_Manager

from PIL import Image

db_manager = Database_Manager(False)
print('! views db created')
#ml_manager = ML_Manager()
print('! views ml created')


# Create your views here.
@api_view(['POST'])
def analyses_post(request):

    print(request.data)
    print(request.FILES)

    print("here analyses_post")
    print(request.data['sample_photo'])
    
    # print('! request.data:', request.data)
    # print('! request.data[sample_photo]:', request.data['sample_photo'])
    # print("! end of data.")

    # Database_Manager.connect_database()
    # serializer = SampleSerializer(data=request.data)
    image, pollenText, pollens = request.data['sample_photo'], request.data['analysis_text'], request.data['pollens']
    # ml_manager.analyze_sample(serializer.data['sample_photo'], "", serializer.data['date'], "John", db_manager)

    # serializer = SampleSerializer(data=request.data)
    # sampleObj = SampleModel(-1,-1, serializer.data['sample_photo'],serializer.data['location_latitude'],serializer.data['location_longitude'],serializer.data['analysis_text'],serializer.data['publication_status']serializer.data['anonymous_status'],serializer.data['pollens'])

    # django image to PIL Image

    if isinstance(image, django.core.files.uploadedfile.InMemoryUploadedFile):
        image = Image.open(image)
    else:
        image = Image.open(image.temporary_file_path())

    # create Sample Model to upload to the database
    sampleObj = SampleModel(-1, 1, image, request.data['date'], request.data['location_latitude'], request.data['location_longitude'], pollenText,
                            request.data['publication_status'], request.data['anonymous_status'], pollens)
    # query database to upload the sample
    result = db_manager.add_sample(sampleObj)
    print('django1', result)
    if result == -1:
        return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result)


@api_view(['GET'])
def analyses_get_by_id(request, pk):
    # Database_Manager.connect_database()
    print('inGet')
    print('django', pk)
    print(type(pk))
    temp = db_manager.get_sample(pk)
    print("bd: ", temp)
    print(temp.sample_id)
    temp2 = Sample(pk,temp.sample_id, temp.academic_id, temp.sample_photo, temp.date, temp.location_latitude,
                    temp.location_longitude,
                    temp.analysis_text, temp.publication_status, temp.anonymous_status, temp.pollens)
    print("aaaa: ",temp2.sample_photo)
    result = SampleSerializer(temp2).data

    #result = serializers.serialize('json',[temp2])
    # temp2 = model_to_dict(temp)
    result = json.dumps(result)
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    #print(result.sample_id)
    #result['sample_photo'] = temp2.sample_photo
    print(result)
    print(temp2.sample_photo)




    # if (result == None):
    #     return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result)
    # analyses = Sample.objects.get(id=pk)


@api_view(['GET'])
def get_all_samples(request):
    
    # Database_Manager.connect_database()
    all_samples = db_manager.get_all_samples()
    samples = []
    print("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
    #print("sample id:,",all_samples[0].sample_id)
    #print("sample id:,", all_samples[1].sample_id)
    for temp in all_samples:
        temp2 = Sample(temp.sample_id,temp.sample_id, temp.academic_id, temp.sample_photo, temp.date, temp.location_latitude,
                    temp.location_longitude,
                    temp.analysis_text, temp.publication_status, temp.anonymous_status, temp.pollens)
        print(temp2.sample_id)
        print("temp:",temp.__str__())
        samples.append(temp2)

    print(samples)
    #test = Sample.objects.all()
    #print(type(test))
    #print(test)
    result = SampleSerializer(samples, many=True).data
    #result = json.dumps(result)

    #print(samples)
    print(result)

    print(len(samples))
    print(len(result))
    #if (result == []):
    #    return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)

    return Response(result)

# def sign_up(request):
#    print(request)
#
#    return HttpResponse(sign_up)

# def profile(request):
#    print(request)

#    return HttpResponse("Profile")

@api_view(['GET'])
def login(request, pk):
    #result = []
    strings = pk.split('~')
    print('a')
    print(request)
    print(pk)
    print(strings)
    #??
    academic = db_manager.get_academic_from_email(strings[0])
    print("academic:")
    print(academic)
    print(academic.password)
    
    AcademicModel
    if academic.password == strings[1]:
        return Response(AcademicSerializer(academic).data)
    else:
        return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
        #return Response()

    #
@api_view(['POST'])
def signup(request):
    print("In sign-up")
    print(request.data)
    print("research")
    #print(request.data['research_gate_link'])
    mdl = AcademicModel(0,request.data['name'],request.data['surname'],request.data['appellation'],
                    request.data['institution'],request.data['job_title'],request.data['email'],request.data['password'],
                    request.data['photo'],request.data['research_gate_link'])

   # mdl = Academic(0,request.data['name'],request.data['surname'],request.data['appellation'],
    #                request.data['institution'],request.data['job_title'],request.data['email'],request.data['password'],
    #                request.data['photo'],request.data['research_gate_link'])

    print("MODEL TEST:")
    print(mdl)
    result = db_manager.add_academic(mdl)
    print("Res:",result)
    if result == -1:
        print("Burada patladi")
        return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
    else:

        return Response(result)

@api_view(['POST'])
def add_feedback(request):
    print(request.data)

    mdl = FeedbackModel(0,request.data['academic_id'],request.data['name'],request.data['email'],request.data['text'],
                        request.data['date'],request.data['status'])
    
    result = db_manager.add_feedback(mdl)
    if result == False:
        return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response(result)


    
#    return HttpResponse("Login info")

# def gmap(request):
#    print(request)

#    return HttpResponse("Map info")
