from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('checkSpam/', views.checkSpam),
    path('output/', views.output_page),
    path('logout/', views.logout),
    path('api_predict', views.api_predict),
    path('insights/', views.insights, name='insights'),
path('feedback/<int:id>/<str:action>/', views.feedback),
]