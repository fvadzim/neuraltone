Установить anaconda 3:
https://www.continuum.io/DOWNLOADS

Установить tensorflow:
```
conda install -c jjhelmus tensorflow=0.12.0rc0
```

Установить pymorphy2:
https://pymorphy2.readthedocs.io/en/latest/user/guide.html#id2
```
pip install pymorphy2[fast]
pip install -U pymorphy2-dicts-ru
pip install -U pymorphy2-dicts-uk
```

Скачать словари nltk:
```
>>> import nltk
>>> nltk.download('all', halt_on_error=False)
```

Запуск сэмпла:
```
python sample.py
```