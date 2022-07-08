#!/usr/bin/env python
# coding: utf-8

# # Análise de sentimento da base do twitter Sentiment140
# 
# ### Dados de Origem
# 
# * http://help.sentiment140.com/for-students
# 
# | sentiment  | id | date | query_string | user | text
# | ---        | -- | -    | -            | -    | ---
# | 0=negativo | -  | -    | -            | -    | the original twitter message
# | 2=neutro   | -  | -    | -            | -    | 
# | 4=positivo | -  | -    | -            | -    |
# 
# 
# ```
# 
# 

# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[ ]:


import pandas as pd
import tensorflow as tf
import numpy as np

tf.random.set_seed(129783)
np.random.seed(3213)


# # Importação
# 
# Dados de origem: 
# * http://help.sentiment140.com/for-students
# * colunas:
#    * sentiment (0=negativo, 2=neutro, 4=positivo)
#    * id
#    * date
#    * query_string
#    * user
#    * text
# 
# Para este exercício:
# * apenas as colunas sentiment e text serão mantidas
# * sentimentos neutros serão descartados
# * sentimentos serão padronizados como 0=negativo e 1=positivo
# 

# In[ ]:


def bloco():
    
    global df_original
    
    df_cols = ['sentiment','id','date','query_string','user','text']

    df_original = pd.read_csv(
        "../input/sentiment140/training.1600000.processed.noemoticon.csv",
        header=None, 
        names=df_cols,
        encoding = "ISO-8859-1"
    )

    df_original.drop(
        ['id','date','query_string','user'],
        axis=1,
        inplace=True
    )
    df_original = df_original[ df_original['sentiment'] != 2 ] 
    df_original['sentiment'] = df_original['sentiment'].apply( lambda x: 1 if x==0 else 0 )
    return df_original

bloco()


# # Padronização (1)
# 
# * Parte 1
#   * remove todas as tags
#   * remove urls
#   * remove identificadores de usuários 
#   * remove caracteres unicode inválidos
#   * remove carcteres não textuais
#   * transforma tudo para minúsculas
# * Parte 2
#   * tokeniza usando o keras
# * Parte 3
#   * separa base de treinamento e de teste

# In[ ]:


#data_limit = 200000
max_words = 100000
max_len = 200

def bloco():
    
    global df_original
    global df_train
    global df_test   
   
    # PARTE 1 - Limpa o texto 
    import re
    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    pat3 = r'<.*?>'
    pat4 = r'&.*?;'
    pat = r'|'.join((pat1, pat2, pat3, pat4))
    def tweet_cleaner(text):       
        text = re.sub(pat,'',text)
        try:
            text = text.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            text = text
        text = re.sub("[^a-zA-Z]", " ", text)
        text = text.lower()
        return text

    df_original['text2'] = df_original['text'].apply( tweet_cleaner )

    
    # PARTE 2 - Tokeniza usando o Keras
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words,lower=True, split=" ")
    tokenizer.fit_on_texts(df_original['text2'])
    df_original['text3'] = tokenizer.texts_to_sequences(df_original['text2'])
    df_original['text4'] = tf.keras.preprocessing.sequence.pad_sequences(
        df_original['text3'], 
        maxlen=max_len,
        #padding='post',
        #truncating='post'        
    ).tolist()                                                                                                   
       
    # PARTE 3 - Separa amostra de treinamento e de teste
   
    # 1 = train | 0 = test
    df_original['rand'] = pd.Series([0,1]).sample(len(df_original), replace=True).array
    
    df_train_sz = 20000
    df_train = df_original[ df_original[ 'rand' ] == 1 ]
    df_train_positive = df_train[ df_train['sentiment'] == 0 ].sample(n=int(df_train_sz/2), replace=True)
    df_train_negative = df_train[ df_train['sentiment'] == 1 ].sample(n=int(df_train_sz/2), replace=True)
    df_train = pd.concat( [ df_train_negative, df_train_positive ] )    
    df_train = df_train.sample(frac=1.0).reset_index(drop=True)
    
    df_test_sz = 40000
    df_test = df_original[ df_original[ 'rand' ] == 0 ].sample(int(df_test_sz/2), replace=True)
    df_test = df_test.sample(frac=1.0).reset_index(drop=True)
    
bloco()
df_train


# Na tabela acima é possível ver os estagios da limpeza:
# * **text** contém o texto original da base 
# * **text2** contém o texto após limpeza de tags, urls, nomes de usuário, números e minusculas
# * **text3** contém o texto codificado em "embeddings" pelo tensorflow. Cada palavra foi convertida em um número. 
# * **text4** contém o texto codificado em "embeddings" com o padding. Só aparecem zeros aqui pois uma coluna com 1,2,3 em text3 será codificada como 0,0,0,0[...],1,2,3. Como é raro encontrar um tweet com mais de 180 palavras, o início é quase sempre [0,0,0,0,...]

# # Modelo

# In[ ]:


model = tf.keras.Sequential()         
model.add(tf.keras.layers.Embedding(max_words, 128))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()


# In[ ]:


import tensorflow_addons as tfa

model.compile(
    #optimizer='rmsprop', 
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=[
        'acc',        
        tf.keras.metrics.Precision(name='prec'),
        tf.keras.metrics.Recall(name='recall'),
        tfa.metrics.F1Score(num_classes=1, threshold=0.5, name='f1')
    ]
)


# In[ ]:


X_train = np.stack( df_train['text4'] )
y_train = df_train['sentiment'].values

print( f'X.shape={X_train.shape} y.shape={y_train.shape}' )


# In[ ]:


my_callbacks = [
        # abandona o processamento se a acurácia não melhorar em até {patitence} épocas
        tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience=6), 
        # grava os modelos intermediários
        tf.keras.callbacks.ModelCheckpoint(filepath='model/model.{epoch:02d}.h5'), # salva o modelo para poder retomar o treinamento
        # grava informações para visualização
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'), 
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.00001)
    ]

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.10,
    callbacks=my_callbacks,
)

print( 'OK' )


# In[ ]:


def bloco():
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(2,1,figsize=(16,6))
    ax[0].plot(history.history['val_loss'], color='#FF0000', label="val_loss")
    ax[0].plot(history.history['loss'], color='#FFA0A0', label="loss")
    legend = ax[0].legend(loc='best', shadow=True)
        
    ax[1].plot(history.history['val_acc'], color='#00FF00', label="val_acc")
    ax[1].plot(history.history['acc'], color='#A0FFA0', label="acc")
    ax[1].set_ylim([0.0,1.0])
    legend = ax[1].legend(loc='best', shadow=True)        
    
bloco()


# In[ ]:


X_test = np.stack( df_test['text4'] )
y_test = df_test['sentiment'].values
y_pred = [ 1 if y_pred > 0.5 else 0 for y_pred in model.predict(X_test).reshape(len(X_test)) ]

df_test['predicted'] = y_pred
df_test[ ['text','sentiment','predicted'] ]


# In[ ]:


def block():
    
    global confusion_mtx
    global confusion_mtx_pc
    
    confusion_mtx = tf.math.confusion_matrix( y_test, y_pred )

    confusion_mtx = pd.DataFrame( confusion_mtx.numpy() )
    confusion_mtx.loc['Total'] = confusion_mtx.sum(numeric_only=True) 
    confusion_mtx['total'] = confusion_mtx[0] + confusion_mtx[1]
    
    confusion_mtx_pc = confusion_mtx / len(y_test)
    
    fp = confusion_mtx.iloc[0,1] 
    fn = confusion_mtx.iloc[1,0]
    tn = confusion_mtx.iloc[0,0]
    tp = confusion_mtx.iloc[1,1] 
    
    total = (tp+tn+fp+fn)
    
    acc       = (tp+tn)/(tp+tn+fp+fn)
    recall    = tp/(tp+fn)
    f1        = (2*acc*recall)/(acc+recall)
    
    fdr  = fp/(fp+tp)
    fnr  = fn/(fn+tp)
    
    tpr = tp/(fn+tp)
    ppv  = tp/(fp+tp)
    
    from IPython.display import display, HTML
    display(HTML(f"""
    
        <style>
           .luc_confusion_mtx td {{ background: white!IMPORTANT; border: 0pt !IMPORTANT; text-align: center!IMPORTANT }}           
           td.luc_confusion_mtx_dp {{ width: 90pt; height: 90pt; background: #c0ffc0!IMPORTANT; border: 1pt solid black!IMPORTANT }} 
           td.luc_confusion_mtx_dn {{ width: 90pt; height: 90pt; background: #ffc0c0!IMPORTANT; border: 1pt solid black!IMPORTANT }}            
        </style>
        
               
        <table class='luc_confusion_mtx'>
        <tr>
            <td></td>
            <td></td>
            <td colspan=2>Previsão</td>
            <td></td>
            <td rowspan=5 style='text-align: left!IMPORTANT'>
                    Acurácia<br><big><big>(TP+TN)/(total)</big></big> = {tp+tn}/{total} = <big>{(tp+tn)*100.0/total:2.1f}%</big><br>
                    <br><br>
                    Considerando que as duas classes (0=sentimento negativo;1=sentimento positivo) tem igual 
                    valor para esta análise, é importante maximizar a diagonal verde / minimizar a diagonal vermelha,
                    portanto os indicadores de Acurácia OU F1 são os mais indicados.<br>
                    <br>
                    Os indicadores de precisão e sensibilidade(recall) podem ser usados em conjunto, mas não são
                    muito intuitivos para este conjunto de dados pois mensuram da perspectiva do "sentimento positivo".
                    Em outras palavras, a sensibilidade indica quantos "sentimentos positivos" corretos foram encontrados
                    e a "precisão" indica do total apontado pelo modelo como "sentimento positivo", quantos eram. 
            </td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td>Negativo</td>
            <td>Positivo</td>
            <td></td>            
        </tr>
        <tr>
            <td rowspan=2>Real</td>
            <td>Negativo</td>
            <td class="luc_confusion_mtx_dp"><big><big>TN</big></big><br>{tn}<br>{tn*100.0/total:2.1f}%</td>
            <td class="luc_confusion_mtx_dn"><big><big>FP</big></big><br>{fp}<br>{fp*100.0/total:2.1f}%</td>
            <td>{tn+fp}</td>
        </tr>        
        <tr>
            <td>Positivo</td>
            <td class="luc_confusion_mtx_dn"><big><big>FN</big></big><br>{fn}<br>{fn*100.0/total:2.1f}%</td>
            <td class="luc_confusion_mtx_dp"><big><big>TP</big></big><br>{tp}<br>{tp*100.0/total:2.1f}%</td>
            <td>{fn+fp}</td>
        </tr>  
        <tr>
            <td></td>
            <td></td>
            <td>{tn+fn}</td>
            <td>{fp+tp}</td>
            <td>{total}</td>
        </tr>  
        </table>    
        
        
       
    """))        
    
    print( f'             Acurácia={acc*100.0:05.2f}% dos apontamentos positivos e negativos estão corretos' )
    print( f' Recall/Sensibilidade={tpr*100.0:05.2f}% dos sentimentos positivos da base foram apontados' )   
    print( f'             Precisão={ppv*100.0:05.2f}% dos sentimentos positivos apontados estão corretos' )    
    print( f'             F1 Score={f1*100.0:05.2f}% média harmônica da acurácia e recall' )        
        
block()


# In[ ]:




