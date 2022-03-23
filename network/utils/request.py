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
