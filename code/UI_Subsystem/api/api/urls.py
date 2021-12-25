from django.urls import path
from .views import analyses, analyses_get_by_id, gmap, login, profile, sign_up, analyses_post, get_all_samples

urlpatterns= [
    path('analysis_posts/',analyses_post),
    path('analysis_posts/<int:pk>/',analyses_get_by_id),
    path('analysis_posts/',get_all_samples)
    #path('/analysis/:id',analyses),
    #path('/sign-up',sign_up),
    #path('/profile',profile),
    #path('/login',login),
    #path('/map',gmap)
    
 ]
