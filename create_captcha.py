from captcha.image import ImageCaptcha
import csv
import random
import os

<<<<<<< HEAD
letters = "0123456789abcdefghijklimnpqrstuvwxyzABCDEFGHIJKLIMNPQRSTUVWXYZ"
# letters = "0123456789"


=======
>>>>>>> 42b37a25b8f448d5026ab0a7903826e8376359b6
def captcha_cs(path,i):#create and save
  img = ImageCaptcha(width=160, height=60, fonts=None, font_sizes=None)

  image = img.generate_image(l1+l2+l3+l4)
  # image.show()
  os.chdir(path)
  image.save(str(i)+'.jpg')

if __name__ == "__main__":
<<<<<<< HEAD
  file = open('prediction_test.csv', 'w')#csv_file_name
=======
  letters = "0123456789abcdefghijklimnpqrstuvwxyzABCDEFGHIJKLIMNPQRSTUVWXYZ"
# letters = "0123456789"

  file = open('train_label.csv', 'w')#csv_file_name
>>>>>>> 42b37a25b8f448d5026ab0a7903826e8376359b6
  writer = csv.writer(file)
  for i in range(0,10000):
    l1 = random.choice(letters)
    l2 = random.choice(letters)
    l3 = random.choice(letters)
    l4 = random.choice(letters)
    writer.writerow([i, l1, l2, l3, l4])
<<<<<<< HEAD
    captcha_cs("/home/cbc106013/deep_learning/captcha/prediction_test",i)#dir_path
  #print(l1+l2+l3+l4)
=======
    captcha_cs("/home/cbc106013/deep_learning/captcha/train_captcha",i)#dir_path
  #print(l1+l2+l3+l4)
>>>>>>> 42b37a25b8f448d5026ab0a7903826e8376359b6
