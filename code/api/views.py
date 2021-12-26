from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .serializers import SampleSerializer

from .Database_Subsytem.SampleModel import SampleModel
from .Database_Subsytem.Database_Manager import Database_Manager
#from .ML_Subsystem.ML_Manager import ML_Manager

db_manager = Database_Manager(False)
#ml_manager = ML_Manager()
# Create your views here.
@api_view(['POST'])
def analyses_post(request):

    print(request.data)
    print(request.data['sample_photo'])
    print("IREMMMMMMMMMMMMMMMMMMMM")

    
    #Database_Manager.connect_database()
    #serializer = SampleSerializer(data=request.data)
    image, pollenText,pollens = request.data['sample_photo'], request.data['analysis_text'], request.data['pollens'] # ml_manager.analyze_sample(serializer.data['sample_photo'], "", serializer.data['date'], "John", db_manager)
    
    #serializer = SampleSerializer(data=request.data)
    #sampleObj = SampleModel(-1,-1, serializer.data['sample_photo'],serializer.data['location_latitude'],serializer.data['location_longitude'],serializer.data['analysis_text'],serializer.data['publication_status']serializer.data['anonymous_status'],serializer.data['pollens'])
    sampleObj = SampleModel(-1, 1, image, request.data['date'], request.data['location_latitude'], request.data['location_longitude'], pollenText, request.data['publication_status'],request.data['anonymous_status'], pollens)
    result = db_manager.add_sample(sampleObj)
    if( result== -1):
         return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def analyses_get_by_id(request,pk):
    # Database_Manager.connect_database()
    result = db_manager.get_sample(pk)

    if(result == None):
            return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result, status=status.HTTP_302_FOUND)
    # analyses = Sample.objects.get(id=pk)

@api_view(['GET'])
def get_all_samples(request):
    # Database_Manager.connect_database()
    result = db_manager.get_all_samples()
    
    if (result == []):
        return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)

    return Response(result, status=status.HTTP_302_FOUND)


# def sign_up(request):
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