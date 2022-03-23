import requests


def post_to_weixi(token, title, name, content):
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                         json={
                             "token": token,
                             "title": title,
                             "name": name,
                             "content": content
                         })
    print(resp.content.decode())


if __name__ == '__main__':
    TOKEN = '9f131c8c711d'
    i = 20
    epoch = 120
    test_recall = [20]
    test_precision = [60]
    post_to_weixi(TOKEN,
                  'Faster R-CNN demo',
                  'AutoDL Linux',
                  'Epoch:{}/{}, R : {:.1f}, P : {:.1f}'.format(i,
                                                 epoch,
                                                 sum(test_recall) / len(test_recall),
                                                 sum(test_precision) / len(test_precision)
                                                )
                  )