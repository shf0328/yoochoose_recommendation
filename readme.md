### Usage
#### Train a model
The following command is used to train a model  
currently support popularity model and item knn model
```
usage: python train.py [-h] [-a ALGORITHM]

optional arguments:
  -h, --help            show this help message and exit
  -a ALGORITHM, --algorithm ALGORITHM
                        choose the recommendation algorithm you want to train
```
Example:
```
# train popularity model 
python train.py -a popularity
```
The output look likes this:
```
you choose to build popularity model...
model built, training time is 51.7425217628479
```

#### Test a model
The following command is used to test a model  
```
usage: python test.py [-h] [-a ALGORITHM] [-k K] [-n NUM] [-s USE_SERVER]

optional arguments:
  -h, --help            show this help message and exit
  -a ALGORITHM, --algorithm ALGORITHM
                        choose the recommendation algorithm you want to train
  -k K                  choose the number of recommended items
  -n NUM, --num NUM     choose the number of test sessions
  -s USE_SERVER, --use_server USE_SERVER
                        choose recommend by server
```
Example:
```
# Using REST API to test item_knn with 5 recommendations and 100 sample sessions, 
python test.py -a item_knn -k 5 -n 100 -s True
```
The output look likes this:
```
you choose to test item_knn model...
model test finished, precision=0.128, recall=0.39263803680981596, throughput=11.700730182557496
```
#### REST API
the service is built on Flask, the file is server.py  
use `python server.py` to run it
  
Example:  
post json data to `SERVER_URL+/model/<model_name>/test`  
And received recommendations list in json format.
``` 
import requests
r = requests.post(SERVER_URL+'/model/{0}/test'.format(model_name), 
            json={
                "session": int(session),
                "source_items": one_group[0:prev_half].to_json(),
                "k": k
            }
    ).json()
```
