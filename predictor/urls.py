"""
URL routes for the predictor app.
"""
from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('analysis/', views.analysis, name='analysis'),
     path('about/', views.about, name='about'),
]