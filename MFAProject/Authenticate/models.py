from django.db import models


# Create your models here.
class VoiceData(models.Model):
    userId = models.CharField(max_length = 5)
    userVoiceEmbedding = models.BinaryField()