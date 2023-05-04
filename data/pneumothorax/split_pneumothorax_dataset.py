import numpy as np
import pydicom
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

def load_image(path, mask_rle):
  ds = pydicom.read_file(path)
  img = ds.pixel_array
  mask = rle2mask(mask_rle, *img.shape[:2]).T

  metadata = {
    'sex': ds.PatientSex,
    'age': ds.PatientAge,
  }

  return img, mask, metadata

def get_df(subset):
  files = glob(f'data/dicom-images-{subset}/*/**/*.dcm')
  file_ids = [os.path.splitext(os.path.basename(x))[0] for x in files]
  files = pd.DataFrame({'ImageId': file_ids, 'path': files})

  data = pd.read_csv(f'data/{subset}-rle.csv')
  data.columns = ['ImageId', 'EncodedPixels']  
  data = pd.merge(data, files, on='ImageId')
  data = data[data['EncodedPixels'] != ' -1']
  data.sort_values('ImageId', inplace=True)
  return data

def main():
  train_df = get_df('train')
  train_df.sample(frac=1, random_state=2022).reset_index(drop=True)
  train_df, valid_df, test_df = np.split(train_df, [int(.8*len(train_df)), int(.9*len(train_df))])

  all_metadata = []
  folders = ['train', 'valid', 'test']
  for folder, df in zip(folders, [train_df, valid_df, test_df]):
    os.makedirs(f'{folder}/input', exist_ok=True)
    os.makedirs(f'{folder}/label', exist_ok=True)
    for index, row in tqdm(df.iterrows(), total=len(df)):
      img, mask, metadata = load_image(row['path'], row['EncodedPixels'])
      all_metadata.append(metadata)
      np.save(f'{folder}/input/{row["ImageId"]}.npy', img)
      np.save(f'{folder}/label/{row["ImageId"]}.npy', mask)
    
  pd.DataFrame(all_metadata).to_csv(f'metadata.csv')

if __name__ == '__main__':
  main()


