#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import pandas as pd
import torch
import re
import random
import nltk
import math

# ================================================
# Автоматическая аугментация текстов и датасетов с использованием 
# предобученной модели Трансформеров, предназначенной для перевода текстов 
# c русского на английский язык
# 
# (c) Artem Kopin 2021
# ================================================

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')




mname = "facebook/wmt19-en-ru"
model_enru = FSMTForConditionalGeneration.from_pretrained(mname)
tokenizer_en = FSMTTokenizer.from_pretrained(mname)
mname = "facebook/wmt19-ru-en"
model_ruen = FSMTForConditionalGeneration.from_pretrained(mname)
tokenizer_ru = FSMTTokenizer.from_pretrained(mname)

# ru_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.ru', tokenizer='moses', bpe='fastbpe')


def translate_ruen(inputs, add_num=1):
    decoded=[]
    input_ids = tokenizer_ru.encode(inputs, return_tensors="pt")
    with torch.no_grad():
        outputs = model_ruen.generate(input_ids, num_beams=add_num, num_return_sequences=add_num)
    # decoded = tokenizer_ru.decode(outputs[0], skip_special_tokens=True)
    
    for o in outputs:
        decoded.append(tokenizer_ru.decode(o, skip_special_tokens=True))
    
    return decoded 


def translate_enru(inputs, add_num=1):
    decoded=[]
    input_ids = tokenizer_en.encode(inputs, return_tensors="pt")
    with torch.no_grad():
        outputs = model_enru.generate(input_ids, num_beams=add_num, num_return_sequences=add_num)
    
    for o in outputs:
        decoded.append(tokenizer_en.decode(o, skip_special_tokens=True))

    # decoded = tokenizer_en.decode(outputs[0], skip_special_tokens=True)
    return decoded 

def augumentation_generator(inputs, multy_factor = 5):
    """
    Artem Kopin
    процедура аугментации одного предложения
    @params:
        inputs            - Required  : входящий текст      
        multy_factor      - Optional  : фактор увеличения
    
    """    
    
    
    pattern = r'[A-Z1-9]{2,9}|[А-Я1-9]{2,9}'
    abb_inp = re.findall(pattern,inputs)
    i=0
    for abb in abb_inp:
        i+=1
        inputs = re.sub(abb, "ABR"+str(i), inputs)
    result = []
    result_check = []
    
    eng_inputs = translate_ruen (inputs, multy_factor)

    for e in eng_inputs:
        
        result_check = translate_enru(e, multy_factor)
        
        for r in result_check:

            if len(re.findall("[а-я]", str(r).lower()))/len(str(r)) > 0:
                #избавляемся от ошибочных переводов
                result.append(r) #translate_enru(e, multy_factor)
    

    result = list(set(result)) #sorted(list(set(result)), reverse=True)  #находим уникальные варианты перевода
    result = random.sample(result, len(result)) #перемешиваем полученные переводы
    
    # Возвращаем все аббревиатуры на место
    for i in range(len(result)):
        abb_res = re.findall(pattern,str(result[i]))
        if len(abb_res)>0:
            for j in range(min(len(abb_res),len(abb_inp))):
                result[i] = str(result[i]).replace(abb_res[j], abb_inp[j])
    
    return result


def augumentation_text_generator(text_inn, multy_factor = 5):
    
    """
    Artem Kopin
    процедура аугментации произвольного длинного текста
    @params:
        text_inn          - Required  : входящий текст      
        multy_factor      - Optional  : сколько разных текстов требуется
    
    """    
    corpus = {}
    text_aug_list = []
    sentences = tokenizer.tokenize (str(text_inn)) 
    
    if len(sentences)>1:
        for sent in sentences:
            corpus[sent] = augumentation_generator(str(sent), int(math.sqrt(multy_factor))+2)
        
        for i in range(multy_factor):
            text = ""
            for sent in sentences:
                text = text + " " + random.choice(corpus[sent])
            text_aug_list.append(text.strip())
    
    elif len(sentences)==1:
        aug_list = augumentation_generator(sentences[0], int(math.sqrt(multy_factor))+2)
        text_aug_list = aug_list[:min(multy_factor,len(aug_list))]
        
    else:
        print("Can't tokenize centences")
    
    return text_aug_list
    
    

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printend = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printend    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), printend)
    # Print New Line on Complete
    if iteration == total: 
        print()

def df_augumentation_generator(df_inn, coll_for_aug, multy_factor=10, merge_style = "down"):
    """
    Artem Kopin
    процедура аугментации DataFrame
    @params:
        df                - Required  : входящий Df
        col_for_aug       - Required  : название колонки для аугментации       
        multy_factor      - Optional  : фактор увеличения
        merge_style       - Optional  : стиль добавления данных "left" - добавление аугументированной колонки слева "down" - добавление в конец "right" - добавление колонки справа
    """
    
    
    coll_augumentated_name = coll_for_aug+"_aug"
    
    addition_text = []
    addition_tags = []
    for index, row in df_inn.iterrows():
        
        add = augumentation_text_generator(row[coll_for_aug], multy_factor)
        addition_text+=add
        addition_tags+=[str(row[coll_for_aug])]*len(add)
        printProgressBar(index+1, len(df_inn), prefix="Augmentation progress: ")
        
    df_add = pd.DataFrame().from_dict({coll_for_aug:addition_tags, coll_augumentated_name:addition_text})
    df = pd.DataFrame()
    if merge_style == "down":
        df_m = pd.merge(df_add, df_inn, how = "left", on = [coll_for_aug])
        df_m = df_m.drop(columns=[coll_for_aug])
        df_m = df_m.rename (columns = {coll_augumentated_name:coll_for_aug})          
        df = pd.concat([df_inn, df_m], ignore_index=True)
    elif merge_style == "left":
        df = pd.merge(df_add[[coll_augumentated_name, coll_for_aug]], df_inn, how = "left", on = [coll_for_aug])
    elif merge_style == "right":
        df = pd.merge(df_inn, df_add, how = "left", on = [coll_for_aug])
    else:
       print ("merge_style is wrong")
        
    return df.drop_duplicates()
 
    
    
class autoaugmentation:
    
    def __init__(self, multy_factor = 5):
        self.multy_factor = multy_factor
        
    def df_generator(self, df_inn, coll_for_aug, merge_style = "down", multy_factor = 5):
        """
        @params:
        df_inn            - Required  : входящий Df
        col_for_aug       - Required  : название колонки для аугментации       
        multy_factor      - Optional  : фактор увеличения
        merge_style       - Optional  : стиль добавления данных "left" - добавление аугументированной колонки слева "down" - добавление в конец "right" - добавление колонки справа
        
        """
        if multy_factor != self.multy_factor:
            self.multy_factor = multy_factor
        else:
            multy_factor = self.multy_factor
        return df_augumentation_generator(df_inn, coll_for_aug, multy_factor, merge_style)
    
    
    def text_generator(self, text_inn, multy_factor = 5):
        """
        @params:
        text_inn          - Required  : входящий текст      
        multy_factor      - Optional  : сколько раз текстов требуется

        """  
        if multy_factor != self.multy_factor:
            self.multy_factor = multy_factor
        else:
            multy_factor = self.multy_factor
        return augumentation_text_generator(text_inn, multy_factor)
    
    
    def translate_ru2en(self, text, multy_factor = 5):
        
        if multy_factor != self.multy_factor:
            self.multy_factor = multy_factor
        else:
            multy_factor = self.multy_factor
        return translate_ruen(text, multy_factor)
    
    
    def translate_en2ru(self, text, multy_factor = 5):
        
        if multy_factor != self.multy_factor:
            self.multy_factor = multy_factor
        else:
            multy_factor = self.multy_factor
        
        return translate_enru(text, multy_factor)
    
    
if __name__ == "__main__":
    ag = autoaugmentation(10)
    ag.text_generator("В составе сборной Англии гол забил Люк Шоу, который отличился на третьей минуте встречи. Но итальянцы все равно одержали победу в финале")
    

