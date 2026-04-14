from django.db import models

# Create your models here.
class Comment(models.Model):
    text = models.TextField()
    result = models.CharField(max_length=20)

    # NEW FIELDS
    confidence = models.FloatField(null=True, blank=True)
    feedback = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)