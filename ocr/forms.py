from django import forms

class UploadForm(forms.Form):
    student_image = forms.ImageField()
    teacher_image = forms.ImageField()
