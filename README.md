hack captcha by cnn with keras
=========================================

# 奕文計畫
![image](https://github.com/cbc106013/captcha_cnn/blob/master/acc94%25.png)

## Reference Material at Issues
<p>captcha.jpg in</p>
<p>https://drive.google.com/drive/folders/1hGFCpwOAyu1PcSXICK2VFgi7f23vX0cx?usp=sharing</p>


2019/01/16(更新)====================================================================
100%
![image](https://github.com/cbc106013/captcha_cnn/blob/master/nptu_captch_kigison%26culdo/train100%25.png)

## 目標資料夾：nptu_captch_kigison&culdo
* DEMO.mp4---------------->DEMO影片
* Demo.py----------------->載入訓練好的模型進行線上預測
* captch_project_test.py-->將原始影像經影像處理去背後另外存檔
* captcha_model.py-------->training code
* data.rar---------------->所有照片(含訓練及測試)
* img2arr.py-------------->將所有照片轉成np.arr以便餵給model
* nptu_captch.csv--------->所有照片的解答(流水號，千位數字，百位數字，十位數字，個位數字)
* nptu_captch.h5---------->此次100%的model存檔
* nptu_captch.npy--------->從照片轉換後的np.arr
* prediction_captcha.py--->載入訓練好的模型進行離線預測
* train100%.png----------->100%截圖

## 製作思路
* dataset----->先人工將300張校務行政系統的驗證碼標記，再將該驗證碼去背另存成一份增加成功率，共600張影像。
* train------->使用不切割形式，以單一模型做四層dense分別判斷對應數字。再使用model.fit調整最佳化。
* prediction-->將驗證碼下載後灰階化直接餵入訓練好的模型進行預測。
* DEMO-------->將prediction改以線上執行

## using: 
* tensorflow-gpu==1.11.0
* keras==2.2.2
* windows 10
