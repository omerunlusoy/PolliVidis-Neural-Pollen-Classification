from django.urls import path
from .views import analyses_get_by_id, analyses_post, analyze, get_academic_by_id, get_all_samples, get_samples_of_academic, login, signup, add_feedback,get_samples_by_filter

urlpatterns= [
    path('analysis_get_id/<int:pk>/', analyses_get_by_id),
    path('analysis_posts/', analyses_post),
    path('analysis_get/', get_all_samples),
    path('login/<str:pk>/',login),
    path('sign-up/',signup),
    path('feedback/', add_feedback),
    path('get_academic_by_id/<int:pk>/',get_academic_by_id),
    path('get_samples_of_academic/<int:pk>/',get_samples_of_academic),
    path('analyze/',analyze),
    path('get_filtered_samples/',get_samples_by_filter)
    #path('/analysis/:id',analyses),
    #path('/profile',profile),
    #path('/map',gmap)  
 ]
