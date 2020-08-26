# Fase 1: Importar las dependencias

**Paper original**: All you need is Attention https://arxiv.org/pdf/1706.03762.pdf


```python
import os
import numpy as np
import pandas as pd
import re
import time

from joblib import dump, load
```


```python
try:
    %tensorflow_version 2.x
except:
    pass
import tensorflow as tf

from tensorflow.keras import layers
import tensorflow_datasets as tfds
```


```python
from mlearner.nlp import Transformer
from mlearner.nlp import Processor_data
```


```python
%load_ext autoreload
%autoreload 2

%matplotlib inline
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    


```python
TRAIN = False
```

# Fase 2: Pre Procesado de Datos



## Carga de Ficheros


```python
with open("data/europarl-v7.es-en.en", 
          mode = "r", encoding = "utf-8") as f:
    europarl_en = f.read()
with open("data/europarl-v7.es-en.es", 
          mode = "r", encoding = "utf-8") as f:
    europarl_es = f.read()
with open("data/P85-Non-Breaking-Prefix.en", 
          mode = "r", encoding = "utf-8") as f:
    non_breaking_prefix_en = f.read()
with open("data/P85-Non-Breaking-Prefix.en", 
          mode = "r", encoding = "utf-8") as f:
    non_breaking_prefix_es = f.read()
```


```python
europarl_en[:230]
```




    'Resumption of the session\nI declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive peri'




```python
europarl_es[:225]
```




    'Reanudación del período de sesiones\nDeclaro reanudado el período de sesiones del Parlamento Europeo, interrumpido el viernes 17 de diciembre pasado, y reitero a Sus Señorías mi deseo de que hayan tenido unas buenas vacaciones'



## Limpieza de datos

Definimos funcion de procesado de texto basada en expresiones regulares


```python
def Function_clean(text):
    
    # Eliminamos la @ y su mención
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    # Eliminamos los links de las URLs
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    return text

```

Se procesan los textos para cada uno de los idiomas:


```python
processor_en = Processor_data(target_vocab_size=2**13, 
                              language="en", 
                              function = Function_clean,
                              name="processor_en",
                             )
processor_es = Processor_data(target_vocab_size=2**13, 
                              language="es", 
                              function = Function_clean,
                              name="processor_es"
                              )
```


```python
if not os.path.isfile("data/corpus_en.csv"):
    corpus_en = europarl_en
    corpus_en = processor_en.clean(corpus_en)
    pd.DataFrame(corpus_en).to_csv("data/corpus_en.csv", index=False)

if not os.path.isfile("data/corpus_es.csv"):
    corpus_es = europarl_es
    corpus_es = processor_es.clean(corpus_es)
    pd.DataFrame(corpus_es).to_csv("data/corpus_es.csv", index=False)

corpus_en = pd.read_csv("data/corpus_en.csv")    
corpus_es = pd.read_csv("data/corpus_es.csv")
```

Exploramos los textos para cada idioma:


```python
corpus_en[0:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resumption of the session</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I declare resumed the session of the European ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
corpus_es[0:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Reanudación del período de sesiones</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Declaro reanudado el período de sesiones del P...</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(corpus_en), len(corpus_es)
```




    (1965735, 1965735)



## Tokenizar el Texto

Tokenizado del texto sin aplicar limpieza (aplicada en el apartado anterior) y sin padding:


```python
if not os.path.isfile('data/processor_en.joblib'):
    tokens_en = processor_en.process_text(corpus_en, 
                                             isclean=True, 
                                             padding=False)
    dump(processor_en, 'data/processor_en.joblib')
else:
    processor_en = load('data/processor_en.joblib')
```


```python
if not os.path.isfile('data/processor_es.joblib'):
    tokens_es = processor_es.process_text(corpus_es, 
                                             isclean=True, 
                                             padding=False)
    dump(processor_es, 'data/processor_es.joblib')
else:
    processor_es = load('data/processor_es.joblib')
```

Tamaño de Vocabulario para los dos idiomas.


```python
if not os.path.isfile("data/inputs.csv") and not os.path.isfile("data/outputs.csv"):
    VOCAB_SIZE_EN = processor_en.tokenizer.vocab_size + 2
    VOCAB_SIZE_ES = processor_es.tokenizer.vocab_size + 2

    print(VOCAB_SIZE_EN, VOCAB_SIZE_ES)
```

Sustituimos los valores NaN con valores vacios:


```python
if not os.path.isfile("data/inputs.csv") and not os.path.isfile("data/outputs.csv"):
    corpus_es = corpus_es.fillna(" ")
    corpus_en = corpus_en.fillna(" ")
```

Preparación de las frases como inputs/outputs del Modelo:

> _**[ \INICIO ]**_ + frase + _**[ \FIN ]**_

- **[ \INICIO ]**: Carácter que determina el inicio de frase.
- **[ \FIN ]**: Carácter que determina el final de frase.


```python
if not os.path.isfile("data/inputs.csv") and not os.path.isfile("data/outputs.csv"):
    inputs = [[VOCAB_SIZE_EN-2] + \
              processor_en.tokenizer.encode(sentence[0]) + [VOCAB_SIZE_EN-1] \
                for sentence in corpus_en.values]

    outputs = [[VOCAB_SIZE_ES-2] + \
               processor_es.tokenizer.encode(sentence[0]) + [VOCAB_SIZE_ES-1] 
                for sentence in corpus_es.values ]
    len(inputs), len(outputs)
```

## Eliminamos las frases demasiado largas


```python
MAX_LENGTH = 20

if not os.path.isfile("data/inputs.csv") and not os.path.isfile("data/outputs.csv"):
    idx_to_remove = [count for count, sent in enumerate(inputs)
                     if len(sent) > MAX_LENGTH]
    if len(idx_to_remove) > 0:
        for idx in reversed(idx_to_remove):
            del inputs[idx]
            del outputs[idx]

    idx_to_remove = [count for count, sent in enumerate(outputs)
                     if len(sent) > MAX_LENGTH]
    if len(idx_to_remove) > 0:
        for idx in reversed(idx_to_remove):
            del inputs[idx]
            del outputs[idx]

    pd.DataFrame(inputs).to_csv("data/inputs.csv", index=False)
    pd.DataFrame(outputs).to_csv("data/outputs.csv", index=False)
```

## Creamos las entradas y las salidas

A medida que entrenamos con bloques, necesitaremos que cada entrada tenga la misma longitud. Rellenamos con el token apropiado, y nos aseguraremos de que este token de relleno no interfiera con nuestro entrenamiento más adelante.


```python
inputs = pd.read_csv("data/inputs.csv").fillna(0).astype(int)   
outputs = pd.read_csv("data/outputs.csv").fillna(0).astype(int)

len(inputs), len(outputs)
```




    (411131, 411131)




```python
MAX_LENGTH = 20

VOCAB_SIZE_EN = 8198
VOCAB_SIZE_ES = 8225 

inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs.values,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=MAX_LENGTH)
outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs.values,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=MAX_LENGTH)
```

Se crea el daset generador para servir los inputs/outputs procesados.


```python
BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
```

# Modelo Transformer - Entrenamiento


```python
tf.keras.backend.clear_session()

# Hiper Parámetros
D_MODEL = 128 # 512
NB_LAYERS = 4 # 6
FFN_UNITS = 512 # 2048
NB_PROJ = 8 # 8
DROPOUT_RATE = 0.1 # 0.1

model_Transformer = Transformer(vocab_size_enc=VOCAB_SIZE_EN,
                          vocab_size_dec=VOCAB_SIZE_ES,
                          d_model=D_MODEL,
                          nb_layers=NB_LAYERS,
                          FFN_units=FFN_UNITS,
                          nb_proj=NB_PROJ,
                          dropout_rate=DROPOUT_RATE)
```

Bucle de entrenamiento


```python
from mlearner.nlp import Transformer_train
```


```python
EPOCHS = 1

Transformer_train(model_Transformer,
                  dataset,
                  d_model=D_MODEL,
                  train=TRAIN,
                  epochs=EPOCHS,
                  checkpoint_path="ckpt/",
                  max_to_keep=5)
```

    The last checkpoint has been restored
    

# Evaluación


```python
def evaluate(inp_sentence):
    inp_sentence = \
        [VOCAB_SIZE_EN-2] + processor_en.tokenizer.encode(inp_sentence) + [VOCAB_SIZE_EN-1]
    enc_input = tf.expand_dims(inp_sentence, axis=0)
    
    output = tf.expand_dims([VOCAB_SIZE_ES-2], axis=0)
    
    for _ in range(MAX_LENGTH):
        predictions = model_Transformer(enc_input, output, False) #(1, seq_length, VOCAB_SIZE_ES)
        
        prediction = predictions[:, -1:, :]
        
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        
        if predicted_id == VOCAB_SIZE_ES-1:
            return tf.squeeze(output, axis=0)
        
        output = tf.concat([output, predicted_id], axis=-1)
        
    return tf.squeeze(output, axis=0)
```


```python
def translate(sentence):
    output = evaluate(sentence).numpy()
    
    predicted_sentence = processor_es.tokenizer.decode(
        [i for i in output if i < VOCAB_SIZE_ES-2]
    )
    
    print("Entrada: {}".format(sentence))
    print("Traducción predicha: {}".format(predicted_sentence))
```

# Predicciones


```python
translate("I have got a house")
```

    Entrada: I have got a house
    Traducción predicha: Yo tengo una casa.
    


```python
translate("This is a problem we have to solve.")
```

    Entrada: This is a problem we have to solve.
    Traducción predicha: Este es un problema que tenemos que resolver.
    


```python
translate("This is a really powerful tool!")
```

    Entrada: This is a really powerful tool!
    Traducción predicha: Esto es realmente un simple edificio.
    


```python
translate("This is an interesting course about Natural Language Processing")
```

    Entrada: This is an interesting course about Natural Language Processing
    Traducción predicha: Es un error interesante sobre la categoría transfronteriza de la carne de la Comunidad.
    


```python

```
