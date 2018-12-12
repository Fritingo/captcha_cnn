import shutil
import requests
import time
SAVEPATH = "/home/cbc106013/deep_learning/captcha/nptu/"
url = 'https://webap.nptu.edu.tw/Web/Modules/CaptchaCreator.aspx?3530'#'http://railway1.hinet.net/ImageOut.jsp'
for i in range(1, 10000):
    response = requests.get(url, stream=True)
    print(response)
    print("=========================================")
    with open(SAVEPATH + str(i) + '.jpg', 'wb') as out_file:

        shutil.copyfileobj(response.raw, out_file)
        print(response.raw)
        print('-----------------------------------------------')
        print(out_file)
        print('###############################################')
    del response
    time.sleep(0.5)
# import requests
#
# import urllib.request
#
# from bs4 import BeautifulSoup
#
# import os
#
# import time
#
#
#
# word = input('Input key word: ')
# url = 'https://www.google.com.tw/search?q='+word+' &rlz=1C1CAFB_enTW617TW621&source=lnms&tbm=isch&sa=X&ved=0ahUKEwienc6V1oLcAhVN-WEKHdD_B3EQ_AUICigB&biw=1128&bih=863'
#
# # url = 'https://www.google.com/search?q=%E7%BE%8E%E5%A5%B3&rlz=1C2CAFB_enTW617TW617&source=lnms&tbm=isch&sa=X&ved=0ahUKEwictOnTmYDcAhXGV7wKHX-OApwQ_AUICigB&biw=1128&bih=960'
#
# photolimit = 30
#
#
#
#
# headers = {'User-Agent': 'Mozilla/5.0'}
#
# response = requests.get(url,headers = headers) #使用header避免訪問受到限制
#
# soup = BeautifulSoup(response.content, 'html.parser')
# print(soup)
# items = soup.find_all('img')
#
#
# folder_path ='./photo/'
#
# if (os.path.exists(folder_path) == False): #判斷資料夾是否存在
#
#     os.makedirs(folder_path) #Create folder
#
#
# for i in range (1,61):
#     for img in soup.select('img'):
#
#         fname = img['title'].split('/')[-1]
#         res2 =requests.get(img['zoomfile'], stream=True)
#         f =open("wow/"+str(i), 'wb')
#         shutil.copyfileobj(res2.raw,f)
#         f.close()
#         del res2
# # for index , item in enumerate (items):
# #
# #     if (item and index < photolimit ):
# #
# #         html = requests.get(item.get('src')) # use 'get' to get photo link path , requests = send request
# #
# #         img_name = folder_path + str(index + 1) + '.png'
# #
# #         # print(index)
# #
# #         with open(img_name,'wb') as file: #以byte的形式將圖片數據寫入
# #
# #             file.write(html.content)
# #
# #             file.flush()
# #
# #         file.close() #close file
# #
# #         print('第 %d 張' % (index + 1))
# #
# #         time.sleep(1)
#
#
#
# print('Done')