from django.db import models

# Create your models here.
class Comment(models.Model):
    text = models.TextField()
    result = models.CharField(max_length=10)