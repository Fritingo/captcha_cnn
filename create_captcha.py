from captcha.image import ImageCaptcha
import csv
import random
import os

letters = "0123456789abcdefghijklimnpqrstuvwxyzABCDEFGHIJKLIMNPQRSTUVWXYZ"



def captcha_cs(path):#create and save
  img = ImageCaptcha(width=160, height=60, fonts=None, font_sizes=None)

  image = img.generate_image(l1+l2+l3+l4)
  # image.show()
  os.chdir(path)
  image.save(l1+l2+l3+l4+'.jpg')

if __name__ == "__main__":
  file = open('test_label.csv', 'w')#csv_file_name
  writer = csv.writer(file)
  for i in range(0,20000):
    l1 = random.choice(letters)
    l2 = random.choice(letters)
    l3 = random.choice(letters)
    l4 = random.choice(letters)
    writer.writerow([l1 + l2 + l3 + l4, l1, l2, l3, l4])
    captcha_cs("/home/cbc106013/deep_learning/captcha_cnn/test_captcha")#dir_path
  #print(l1+l2+l3+l4)