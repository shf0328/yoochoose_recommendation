import requests
r = requests.post('http://127.0.0.1:6789'+'/model/{0}/test'.format('item_knn'),
                  json={
                      "session": 0,
                      "source_items": '{"Item_ID":{"0":214530776,"1":214530776}}',
                      "k": 5
            }
    ).json()
print(r)  # [214553857, 214530774, 214840530, 214840524, 214551504]
