import os
import requests
from bs4 import BeautifulSoup
import shutil
import numpy as np
import time

os.mkdir('new')
for i in range(1,3,1):
    title_name = 'https://www.jkforum.net/forum-574-'+str(i)+'.html'
    # print(title_name)

main_web = requests.get(title_name)
soup_first = BeautifulSoup(main_web.text)
# print(soup_first)

test = []

for q in soup_first.select('body'):
    for a in q.select('div'):
        for w in a.select('div'):
            for e in w.select('form'):
                for r in e.select('ul'):
                    for t in r.select('li'):
                        for y in t.select('a'):
                            # print(y.get('href'))
                            test.append(y.get('href'))
for i in range (0,176,4):
    web_name = 'https://www.jkforum.net/'+test[i]

    res = requests.get(web_name)
    soup = BeautifulSoup(res.text)
    a= os.getcwd()
    print(a)
    time.sleep(5)


    for img in soup.select('.zoom'):
    #    fname =  img

        fname = img['title'].split('/')[-1]
        res2 =requests.get(img['zoomfile'], stream=True)
        f =open("new/"+fname, 'wb')
        shutil.copyfileobj(res2.raw,f)
        f.close()
        del res2
        # time.sleep(10)
#======================================================================
    # print('https://www.jkforum.net/'+test[i])


    # print(web.get('href'))
#
#     # fname = img['title'].split('/')[-1]
#     res2 =requests.get(img['title'], stream=True)
#     print(res2)
#     # f =open("new/"+fname, 'wb')
#     # shutil.copyfileobj(res2.raw,f)
#     f.close()
#     del res2


# res = requests.get('https://www.jkforum.net/thread-8739907-1-1.html')
# soup = BeautifulSoup(res.text)
# a= os.getcwd()
# print(a)
# # os.mkdir('new')
#
# for img in soup.select('.zoom'):
# #    fname =  img
#
#     fname = img['title'].split('/')[-1]
#     res2 =requests.get(img['zoomfile'], stream=True)
#     f =open("new/"+fname, 'wb')
#     shutil.copyfileobj(res2.raw,f)
#     f.close()
#     del res2

