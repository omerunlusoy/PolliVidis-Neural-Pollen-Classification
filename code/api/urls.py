from django.urls import path
from .views import analyses_get_by_id, analyses_post, get_all_samples, login, signup

urlpatterns= [
    path('analysis_get_id/<int:pk>/', analyses_get_by_id),
    path('analysis_posts/', analyses_post),
    path('analysis_get/', get_all_samples),
    path('login/<any:pk>',login),
    path('sign-up/',signup)
    #path('/analysis/:id',analyses),
    #path('/profile',profile),
    #path('/map',gmap)
    
 ]
