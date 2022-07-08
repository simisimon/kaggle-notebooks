#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import clear_output


# **The first time you start your notebook, start the cell below and then click on it when the <code>RESTART RUNTIME</code> button appears (or restart the environment manually through Runtime Environment->Restart Runtime)** <br />
# Then run each cell in turn and go to the last one (cell "Results")
# <br /> <br />
# **При первом запуске ноутбука запустите ячейку ниже, а затем при появлении кнопки <code>RESTART RUNTIME</code> нажмите на нее(или же перезапустите среду вручную через Среда выполнения->Перезапустить среду выполнения)** <br />
# Потом по очереди запустите каждую из ячеек и перейдите к последней(ячейка "Results")

# In[ ]:


get_ipython().system('pip3 install simpledemotivators')


# ## Install environment

# In[ ]:


get_ipython().system('pip3 install transformers==2.8.0')
clear_output()


# In[ ]:


get_ipython().system('pip3 install urllib3==1.25.4')
get_ipython().system('wget https://raw.githubusercontent.com/sberbank-ai/ru-gpts/master/pretrain_transformers.py')
get_ipython().system('wget https://raw.githubusercontent.com/sberbank-ai/ru-gpts/master/generate_transformers.py')
clear_output()


# In[ ]:


get_ipython().system('git clone https://github.com/NVIDIA/apex')
get_ipython().system('cd apex; python setup.py install')
clear_output()


# Loading font from github

# In[ ]:


get_ipython().system('git clone https://github.com/Egoluback/neurowolf.git')


# # Preprocessing data

# In[ ]:


import pandas as pd

data = pd.read_csv('/kaggle/input/russian-gang-quotes/quotes.csv')

lines = data["0"].tolist()


# In[ ]:


with open("train.txt", "w+") as file:
    file.writelines(lines)


# # Train and tests

# ## Train NLP model

# In[ ]:


import torch

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# We'll fine-tune small gpt3 model(based on gpt2) from SberAI <br />
# Training lasts no longer than two minutes
# <br /> <br />
# Файнтюним маленькую gpt3 модель(основанную на gpt2) от сбера <br />
# Обучение длится не дольше двух минут

# In[ ]:


get_ipython().system('python pretrain_transformers.py      --output_dir=model      --model_type=gpt2      --model_name_or_path=sberbank-ai/rugpt3small_based_on_gpt2      --do_train      --train_data_file=train.txt      --do_eval      --fp16      --eval_data_file=valid.txt      --per_gpu_train_batch_size 1      --gradient_accumulation_steps 1      --num_train_epochs 5      --block_size 2048      --overwrite_output_dir')


# ## Save model

# In[ ]:


# so far no need
# !cp -r model/  path-to-model


# ## Load model

# In[ ]:


from transformers import GPT2Tokenizer, GPT2LMHeadModel

path = "model"

tokenizer_gen = GPT2Tokenizer.from_pretrained(path)
model_gen = GPT2LMHeadModel.from_pretrained(path)
model_gen.to("cuda")


# In[ ]:


import copy

bad_word_ids = [
    [203], # \n
    [225], # weird space 1
    [28664], # weird space 2
    [13298], # weird space 3
    [206], # \r
    [49120], # html
    [25872], # http
    [3886], # amp
    [38512], # nbsp
    [10], # &
    [5436], # & (another)
    [5861], # http
    [372], # yet another line break
    [421, 4395], # МСК
    [64], # \
    [33077], # https
    [1572], # ru
    [11101], # Источник
]

def gen_fragment(context, bad_word_ids=bad_word_ids, print_debug_output=False, 
                 temperature=1.0, max_length=20, min_length=5):
    while True:
        input_ids = tokenizer_gen.encode(context, add_special_tokens=False, 
                                    return_tensors="pt").to("cuda")
        input_ids = input_ids[:, -1700:]
        input_size = input_ids.size(1)

        output_sequences = model_gen.generate(
            input_ids=input_ids,
            max_length=max_length + input_size,
            min_length=min_length + input_size,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            temperature=1.0,
            pad_token_id=0,
            eos_token_id=2,
            bad_words_ids=bad_word_ids,
            no_repeat_ngram_size=10
        )

        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()
            
        generated_sequence = output_sequences[0].tolist()[input_size:]
        if print_debug_output:
            for idx in generated_sequence:
                print(idx, tokenizer_gen.decode([idx], clean_up_tokenization_spaces=True).strip())
        text = tokenizer_gen.decode(generated_sequence, clean_up_tokenization_spaces=True)
        if len(text) == 0 or text.rfind(".") == -1: continue
        text = text[: text.rfind(".") + 1]
        return context + text


# ## Generating demotivators

# In[ ]:


from simpledemotivators import Demotivator
from PIL import Image

import requests, re

def create_picture(start, picture_target="wolf"):
    r = requests.post(
        "https://api.deepai.org/api/text2img",
        data={'text': picture_target},
        headers={'api-key': '83bfdbfd-e539-4f7e-be8c-3a49325f8cac'}
    )

    url_image = r.json()['output_url']

    text = gen_fragment(start).replace("»", '').replace("«", '').replace('"', '').replace('...', '.').replace('!', '.').replace('?', '.')
    sentences = text.split('.')
    

    if len(sentences) > 1:
        dem = Demotivator(sentences[0], sentences[1])
    else:
        dem = Demotivator(sentences[0])

    dem.create(url_image, use_url=True, font_name='/kaggle/working/neurowolf/data/tnr.ttf')


# In[ ]:


begin_phrases = ['Своих друзей',
                'Спасибо Богу',
                'Это моя жизнь',
                'Слово пацана',
                'Пацан сказал',
                'Достойная девушка не',
                'Не прощай',
                'Сука, пацаны, берегите',
                'Чётко - это когда',
                'Брат - это тот',
                'За братву',
                'Моя братва',
                'Настоящий пацан',
                'Пацан не станет',
                'Быть пацаном - значит',
                'Как жаль, что я живу в стране,',
                 "Вот это мужик, а вы, девчонки, и дальше"]


# # Results

# Generation of a random "gang" quote, using pre-prepared "inoculum" <br />
# As a query image - wolf
# <br /> <br />
# Генерация случайной пацанской цитаты, используя подготовленные заранее "затравки" <br />
# В качестве запроса изображения - волк

# In[ ]:


from random import choice

create_picture(choice(begin_phrases))
Image.open('demresult.jpg')


# Generation of a custom quote (you can enter the margin yourself) <br />
# The query for the image is also specified by yourself (parameter picture_target)
# <br /> <br />
# Генерация кастомной пацанской цитаты(можно ввести задел самому) <br />
# Запрос для изображения указывается также самостоятельно(параметр picture_target)
# 

# In[ ]:


phrase_begin = "Шоколад ни в чем не виноват, "
create_picture(phrase_begin, picture_target="chocolate")
Image.open('demresult.jpg')

