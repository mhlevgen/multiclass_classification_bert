## Multilabel text classification

1. Data file you can find in `src/data` folder. There is multiclass classification problem
with 4 classes in total. Classes are balanced.
2. I trained 2 models: baseline (tf-idf + catboost) and advanced (bert). Training pipelines in 
`src/train_tfidf_catboost.ipynb` and `src/train_bert.ipynb` notebooks. 
Both models trained on same train data and evaluated on the same test data.

3. Confusion matrix catboost: 

<img alt="tfidf" src="./tfidf.png" width="300"/>

4. Confusion matrix bert:

<img alt="tfidf" src="./bert.png" width="300"/>

5. Tf idf + catboost has 11 misclassified objects and bert 5 out of 2000 test samples. That was quite predictable.
6. I implemented API on flask with catboost model. To launch in use docker container:

`docker build . -t bert:latest`

`docker run -it -p 8889:8889 bert:latest`

6. Request API according to this example

```
curl --request POST 'http://localhost:8889/get-category' \
--data-raw '{
    "main_text": <some text>,
    "add_text": <some text>,
    "manufacturer": <some text>,
}'
```
