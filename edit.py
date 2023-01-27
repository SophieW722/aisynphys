import os, sys, base64, pickle, json

url_file = 'download_urls'
tmp_file = url_file + '.tmp'


def check_json(filename):
    try:
        urls = json.load(open(tmp_file, 'r'))
    except Exception as exc:
        sys.excepthook(*sys.exc_info())
        return False
    return True



urls = pickle.loads(base64.b64decode(open(url_file, 'r').read()))
js = json.dumps(urls, indent='    ')
open(tmp_file, 'w').write(js)

while True:
    os.system('nano ' + tmp_file)
    if check_json(tmp_file):
        break
    input("Press enter to edit again..")
    

urls = json.load(open(tmp_file, 'r'))
b64p = base64.b64encode(pickle.dumps(urls))
open(url_file, 'wb').write(b64p)
os.remove(tmp_file)
