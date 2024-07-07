from django.shortcuts import render
from .forms import UploadForm
import requests
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import os

def image_to_text(student_image_path, teacher_image_path, api_key):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': api_key,
        'OCREngine': 2,
        'language': 'eng',
    }
    with open(student_image_path, 'rb') as student_image_file:
        student_files = {'file': (student_image_path, student_image_file, 'image/jpeg')}
        student_response = requests.post(url, files=student_files, data=payload)
    student_result = student_response.json()
    student_text = student_result['ParsedResults'][0]['ParsedText'] if student_result['ParsedResults'] else 'No text found'
    with open(teacher_image_path, 'rb') as teacher_image_file:
        teacher_files = {'file': (teacher_image_path, teacher_image_file, 'image/jpeg')}
        teacher_response = requests.post(url, files=teacher_files, data=payload)
    teacher_result = teacher_response.json()
    teacher_text = teacher_result['ParsedResults'][0]['ParsedText'] if teacher_result['ParsedResults'] else 'No text found'
    
    return student_text, teacher_text

def answer_summary(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, early_stopping=False)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def keyword_extraction(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    extracted_keywords = filtered_tokens[:5]  # Extracting first 5 non-stopword tokens as example
    return extracted_keywords

def scoring(extracted_keywords, answer_key_keywords, max_marks):
    score = len(set(extracted_keywords) & set(answer_key_keywords)) / len(answer_key_keywords) * max_marks
    return score

def index(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            student_image = request.FILES['student_image']
            teacher_image = request.FILES['teacher_image']
            api_key = 'K87835079188957'
            student_image_path = os.path.join('uploads', student_image.name)
            teacher_image_path = os.path.join('uploads', teacher_image.name)

            with open(student_image_path, 'wb+') as destination:
                for chunk in student_image.chunks():
                    destination.write(chunk)

            with open(teacher_image_path, 'wb+') as destination:
                for chunk in teacher_image.chunks():
                    destination.write(chunk)

            student_text, teacher_text = image_to_text(student_image_path, teacher_image_path, api_key)
            student_summary = answer_summary(student_text)
            student_keywords = keyword_extraction(student_summary)
            teacher_summary = answer_summary(teacher_text)
            teacher_keywords = keyword_extraction(teacher_summary)
            max_marks = 10  
            student_score = scoring(student_keywords, teacher_keywords, max_marks)
            
            return render(request, 'result.html', {
                'student_text': student_text, 'teacher_text': teacher_text,
                'student_summary': student_summary, 'teacher_summary': teacher_summary,
                'student_keywords': student_keywords, 'teacher_keywords': teacher_keywords,
                'student_score': student_score, 'max_marks': max_marks,
            })
    else:
        form = UploadForm()
    
    return render(request, 'index.html', {'form': form})
