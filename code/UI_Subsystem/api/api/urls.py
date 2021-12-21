from django.urls import path
from .views import analyses, gmap, login, profile, sign_up

urlpatterns= [
    path('/analysis/:id',analyses),
    path('/sign-up',sign_up),
    path('/profile',profile),
    path('/login',login),
    path('/map',gmap)
 ]
