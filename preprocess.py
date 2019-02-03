import glob
import numpy as np
from PIL import Image
import pandas

path = '.\\Test\\*.jpg'
files = glob.glob(path)
col_names =  ['image', 'age', 'gender', 'race']
df = pandas.DataFrame(columns = col_names)

count = 0
 
for file_name in files:
  img = Image.open(file_name)
  img = img.resize((32, 32),Image.ANTIALIAS)
  img = np.asarray(img, dtype=np.float32) / 255
  file_name = file_name.split('\\')[-1]
  file_name = file_name.split('.')[0]
  file_name = file_name.split('_')
  my_dic = dict()
  if len(file_name) < 4:
    continue
  my_dic['img'] = np.array(img).reshape(-1,)
  my_dic['age'] = np.zeros(10)
  my_dic['age'][min(9,int(int(file_name[0]) / 10))] = 1
  my_dic['gender'] = np.zeros(2)
  my_dic['gender'][int(file_name[1])] = 1
  my_dic['race'] = np.zeros(5)
  my_dic['race'][int(file_name[2])] = 1
  df.loc[len(df)] = my_dic

  count = count + 1
  print(my_dic['img'])
  
  
df.to_pickle('UTKFace.pkl')
print(df.head())