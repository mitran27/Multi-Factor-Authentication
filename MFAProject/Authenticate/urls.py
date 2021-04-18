from django.urls import path
from . import views


urlpatterns = [
    path('RegisterPage',views.registerpage,name = 'RegisterPage'),
    path('Register',views.Register,name='Register'),
    path('login',views.login,name="login"),
    path('loginPage',views.loginPage,name="loginpage"),    
    path('Authenticate',views.authenticate , name = "video"),
    path('voiceRegister',views.voiceRegister , name = "register"),
    path('welcome',views.welcome,name="welcome"),
]

