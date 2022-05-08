import django
from django.forms import model_to_dict
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.core import serializers
import json
from firebase_admin import credentials, initialize_app, storage, delete_app, get_app
import matplotlib.pyplot as plt
import datetime
from .local_paths import get_firebase_path
from api.Database_Subsytem.AcademicModel import AcademicModel

from .models import Sample, Academic
from .serializers import AcademicSerializer, SampleSerializer,FeedbackSerializer

from .Database_Subsytem.SampleModel import SampleModel
from .Database_Subsytem.Database_Manager import Database_Manager
from .Database_Subsytem.FeedbackModel import FeedbackModel
from .ML_Subsystem.ML_Manager import ML_Manager
#from .ML_Subsystem.ML_Manager import ML_Manager

from PIL import Image

cred = credentials.Certificate(get_firebase_path())

try:
    app = initialize_app(cred, {'storageBucket': 'fir-react1-70dd6.appspot.com'})
except:
    app = get_app()
    delete_app(app)
    print("App deleted")
    app = initialize_app(cred, {'storageBucket': 'fir-react1-70dd6.appspot.com'})




#firebase = Firebase(firebaseConfig)
#initialize_app(cred,firebaseConfig)
#strg = firebase.

#storage = firebase.storage()

#fb_storage = firebase_storage.
#bucket = firebase_admin.storage().bucket();
db_manager = Database_Manager(False)
ml_manager = ML_Manager(load_model=True)
print('! views db created')
#ml_manager = ML_Manager()
print('! views ml created')


# Create your views here.
@api_view(['POST'])
def analyses_post(request):

    #print(request.data)
    #print(request.FILES)

    print("here analyses_post")
    #print(request.data['sample_photo'])
    
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
    sampleObj = SampleModel(-1, request.data['academic_id'], image, request.data['date'], request.data['location_latitude'], request.data['location_longitude'], pollenText,
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
    #print(type(pk))
    temp = db_manager.get_sample(pk)
    #print("bd: ", temp)
    print(temp.sample_id)
    temp2 = Sample(pk,temp.sample_id, temp.academic_id, temp.sample_photo, temp.date, temp.location_latitude,
                    temp.location_longitude,
                    temp.analysis_text, temp.publication_status, temp.anonymous_status, temp.pollens)
    print("aaaa: ",temp2.sample_photo)
    result = SampleSerializer(temp2).data

    #result = serializers.serialize('json',[temp2])
    # temp2 = model_to_dict(temp)
    result = json.dumps(result)
    #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    #print(result.sample_id)
    #result['sample_photo'] = temp2.sample_photo
    #print(result)
    #print(temp2.sample_photo)




    # if (result == None):
    #     return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result)
    # analyses = Sample.objects.get(id=pk)


@api_view(['GET'])
def get_all_samples(request):
    print("got all samples")

    # Database_Manager.connect_database()
    all_samples = db_manager.get_all_samples()
    samples = []
    #print("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
    #print("sample id:,",all_samples[0].sample_id)
    #print("sample id:,", all_samples[1].sample_id)
    for temp in all_samples:
        if temp.academic_id == 1:
                continue
        temp2 = Sample(temp.sample_id,temp.sample_id, temp.academic_id, temp.sample_photo, temp.date, temp.location_latitude,
                    temp.location_longitude,
                    temp.analysis_text, temp.publication_status, temp.anonymous_status, temp.pollens)
        print(temp2.sample_id)
        #print("temp:",temp.__str__())
        samples.append(temp2)

    #print(samples)
    #test = Sample.objects.all()
    #print(type(test))
    #print(test)
    result = SampleSerializer(samples, many=True).data
    #result = json.dumps(result)

    #print(samples)
    #print(result)

    #print(len(samples))
    #print(len(result))
    #if (result == []):
    #    return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)

    return Response(result)


@api_view(['POST'])
def get_samples_by_filter(request):
    month_dict = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06", "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}

    ## Request: api url
    ## Request: data
    pollens = request.data['pollens']
    start_date = request.data['startDate'] # null if not specified
    end_date = request.data['endDate'] # null if not specified

    temp_start = start_date.split(" ")
    start_date = month_dict[temp_start[1]] + "-" + temp_start[2] + "-" + temp_start[3]
    start_date = datetime.datetime.strptime(start_date, '%m-%d-%Y').date()

    temp_end = end_date.split(" ")
    end_date = month_dict[temp_end[1]] + "-" + temp_end[2] + "-" + temp_end[3]
    end_date = datetime.datetime.strptime(end_date, '%m-%d-%Y').date()

    if start_date == end_date:
        start_date = None
        end_date = None
        #print("agubugu!!!!!")
    #print("pollens",pollens)
    #print(start_date)
    #print(end_date)

    if pollens == []:
        return get_all_samples()
    else:
        # Database_Manager.connect_database()
        all_samples = db_manager.get_all_samples()
        samples = []
        
        for temp in all_samples:
            #print("abv", temp.date)
            if temp.date == "":
                continue
            temp3 = temp.date.split(" ")
            if "-" in temp3[0]:
                temp_date = temp3[0]
                temp_date = datetime.datetime.strptime(temp_date, '%Y-%m-%d').date()
            else:
                temp_date = month_dict[temp3[1]] + "-" + temp3[2] + "-" + temp3[3]
                temp_date = datetime.datetime.strptime(temp_date, '%m-%d-%Y').date()

            if temp.academic_id == 1:
                #print("tem.sampleid")
                continue
            
            c = 0
            if temp.pollens == []:
                #print("boş bu")
                continue
            

            #print("temp:" , temp_date)
            for pollen in temp.pollens:
                #print("test_pollen",pollen)
                #print(type(pollen))
                if pollen in pollens:
                    #print("pollen buldu:", pollen)
                    c = 1
                    break

            if c == 0:
                #print("c == 0")
                continue
            

            if start_date != None and end_date != None:
                #print("if 1")
                #print(type(start_date))
                #print(type(temp.date))

                if temp_date < start_date or temp_date > end_date:
                    #print("if 2")
                    continue
            
            temp2 = Sample(temp.sample_id,temp.sample_id, temp.academic_id, temp.sample_photo, temp.date, temp.location_latitude,
                        temp.location_longitude,
                        temp.analysis_text, temp.publication_status, temp.anonymous_status, temp.pollens)
            print(temp2.sample_id)
            #print("temp:",temp.__str__())

            samples.append(temp2)

        #print("samples:" ,samples)
        #test = Sample.objects.all()
        #print(type(test))
        #print(test)
        
        result = SampleSerializer(samples, many=True).data
        #result = json.dumps(result)

        #print(samples)
        #print(result)

        #print(len(samples))
        #print(len(result))
        
        return Response(result)


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

@api_view(['GET'])
def get_academic_by_id(request,pk):
    print(pk)

    temp = db_manager.get_academic_from_id(pk)

    temp2 = Academic(pk,temp.academic_id, temp.name, temp.surname, temp.appellation,temp.institution, temp.job_title,
                    temp.email,
                    temp.password, temp.photo, temp.research_gate_link)
    result = AcademicSerializer(temp2).data
    print(result)
    result = json.dumps(result)

    print(result)

    return Response(result)

@api_view(['GET'])
def get_samples_of_academic(request,pk):
    print(pk)
    all_samples = db_manager.get_samples_of_academic(pk)
    samples = []
    for temp in all_samples:
        temp2 = Sample(temp.sample_id,temp.sample_id, temp.academic_id, temp.sample_photo, temp.date, temp.location_latitude,
                    temp.location_longitude,
                    temp.analysis_text, temp.publication_status, temp.anonymous_status, temp.pollens)
        #print(temp2.sample_id)
        #print("temp:",temp.__str__())
        samples.append(temp2)
    result = SampleSerializer(samples, many=True).data
    return Response(result)

@api_view(['PUT'])
def analyze(request):

    print("analyze enter")
    photo_url = request.data['url']
    photo_id = request.data['id']
    morp = request.data['morp']
    fileName = 'files/' + str(photo_id)
    
    print("analyze init")
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    fileName2 = "./"+str(photo_id) +"_.jpg"
    
    blob.download_to_filename(fileName2)

    print("img downlaod complete")
    sample_image = Image.open(fileName2)

    print("ml start")
    source_img, analysis_text, pollens_dict = ml_manager.analyze_sample(sample_image, location=None, date=None, academic_name=None, morphology_sequence=morp,
                                                                        test_extraction=False)
    analysis_text = analysis_text.replace("/n","\n")
    #analysis_text = "<p>" + analysis_text
    #analysis_text = analysis_text.replace("\n","</p><p>")
    #analysis_text = analysis_text[:-3]
    print("ml finish")
    fileName2 = str(photo_id) + "_final.jpg"
    fileName = 'files/' + fileName2
    blob = bucket.blob(fileName)
    img = source_img.save(fileName2)
    blob.upload_from_filename(fileName2)
    blob.make_public()

    smpl = db_manager.get_sample(photo_id)
    smpl.analysis_text = analysis_text
    
    smpl.pollens = pollens_dict
    print("analysis_text")
    print(analysis_text)
    print("sample:")
    print(smpl)
    db_manager.update_sample(smpl)

    for pollen_name in smpl.pollens:
        print("pollen_name:", pollen_name)
        print("photo_id:", photo_id)
        print("count:",smpl.pollens[pollen_name])
        db_manager.add_pollen_has(photo_id,pollen_name,smpl.pollens[pollen_name])
         
    #db_manager.delete_sample(photo_id)
    #db_manager.add_sample(smpl)
    return Response(True)

@api_view(['DELETE'])
def remove_analysis(request,pk):
    print("irem:",pk)
    res = db_manager.delete_sample(pk)
    return Response(res)
    #print(request)

