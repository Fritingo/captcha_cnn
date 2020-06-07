from keras.models import load_model
import numpy as np
from PIL import Image
from selenium.webdriver import Chrome
from PIL import Image
import io
from time import sleep

account = "account"
password = "password"

letters = "0123456789"

browser = Chrome("chromedriver.exe") # chromedriver 路徑
browser.set_window_size(1080,720)
browser.get('https://webap.nptu.edu.tw/Web/Secure/default.aspx')

model = load_model('nptu_captch.h5')

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


for i in range(10):  #predict 份數
    browser.find_element_by_id("LoginDefault_ibtLoginStd").click()
    captcha_img = browser.find_element_by_id('imgCaptcha')
    set_CheckCode = browser.find_element_by_id('LoginStd_txtCheckCode')
    set_account = browser.find_element_by_id('LoginStd_txtAccount')
    set_password = browser.find_element_by_id('LoginStd_txtPassWord')
    captcha_img.click()
    set_CheckCode.clear()
    img = Image.open(io.BytesIO(captcha_img.screenshot_as_png))
    # img = img.convert("RGB")
    # img = ig.resize((95, 35))
    # img.show()
    prediction_data = np.stack(np.array(img)/255.0)  #predict img local
    prediction_data=rgb2gray(prediction_data) #灰階
    prediction_data= prediction_data.reshape(-1,35,95,1)
    prediction = np.array(model.predict(prediction_data))

    ans = ""
    p_max = max(prediction[0,0,:])
    for j in range(4):
        p_max = max(prediction[j, 0, :])
        for i in range(len(letters)):
            if (prediction[j,0,i] == p_max ):
                ans += letters[i]
                break
    print(ans)


    set_account.send_keys(account)
    set_password.send_keys(password)
    sleep(1)
    set_CheckCode.send_keys(ans)
    sleep(0.2)
    browser.find_element_by_id("LoginStd_ibtLogin").click()
    sleep(0.5)

    browser.switch_to.frame(browser.find_element_by_css_selector("frame[name='MAIN'"))
    browser.find_element_by_id('CommonHeader_ibtLogOut').click()
    browser.switch_to.alert.accept()

    sleep(1)

browser.quit()



