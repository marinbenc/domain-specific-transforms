import sys
import os
import os.path as p

import numpy as np
import cv2 as cv

import pydicom as dicom
from glob import glob

import matplotlib.pyplot as plt

from skimage.registration import phase_cross_correlation

sys.path.append('../../')
import shutil

labels_folder = 'data/label'
inputs_folder = 'data/dicom_input'
peri_folder = 'data/peri'

train, valid, test = (14, 1, 5)
patients = np.array(os.listdir(labels_folder))
patients.sort()

np.random.seed(42)
np.random.shuffle(patients)

train_patients = patients[:train]
valid_patients = patients[train:train + valid]
test_patients = patients[-test:]

assert(len(train_patients) + len(valid_patients) + len(test_patients) == len(patients))

folders = {
  'train': train_patients,
  'valid': valid_patients,
  'test': test_patients
}

print(folders)

for (folder, folder_patients) in folders.items():
  os.makedirs(p.join(folder, 'input'), exist_ok=True)
  os.makedirs(p.join(folder, 'label'), exist_ok=True)

  for patient in folder_patients:
    patient_labels = os.listdir(p.join(labels_folder, patient))
    patient_labels.sort()

    for label_file in patient_labels:
      print(f'{patient}_{label_file}')
      label_eat = cv.imread(p.join(labels_folder, patient, label_file), cv.IMREAD_GRAYSCALE)

      if not p.exists(p.join(peri_folder, patient, label_file)):
        print(f'No peri file for {patient}_{label_file}')
        continue

      label_peri = cv.imread(p.join(peri_folder, patient, label_file), cv.IMREAD_GRAYSCALE)

      slice_id = label_file.replace('.png', '')
      input_file = glob(p.join(inputs_folder, patient, f'*{slice_id}_an.dcm'))
      if len(input_file) == 0:
        print(f'No input file for {patient}_{label_file}')
        continue
      input_file = input_file[0]
      dcm = dicom.dcmread(input_file)
      # to Hounsfield
      input = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
      
      #input = cv.imread(p.join(inputs_folder, patient, label_file), cv.IMREAD_GRAYSCALE)

      size = (512, 512)
      input = cv.resize(input, dsize=size, interpolation=cv.INTER_CUBIC)
      label_eat = cv.resize(label_eat, dsize=size, interpolation=cv.INTER_NEAREST)
      label_peri = cv.resize(label_peri, dsize=size, interpolation=cv.INTER_NEAREST)

      # labels are shifted and not aligned with DCM inputs
      # use cross correlation to align them
      # calculate shift on eat label and then appy to peri label
      input_normal = input.copy()
      # threshold to fat range to avoid noise
      input_normal[input_normal < -140] = -140
      input_normal[input_normal > -30] = -140
      input_normal = (input_normal - input_normal.min()) / (input_normal.max() - input_normal.min() + 1e-8)
      label_normal = (label_eat - label_eat.min()) / (label_eat.max() - label_eat.min() + 1e-8)
      # calculate shift from label to input
      shift, error, diff = phase_cross_correlation(input_normal, label_normal)
      if np.abs(diff) > 0.1:
        print('Warning: large diff ' + str(diff) + ' for ' + patient + '_' + label_file)
        continue
      M = np.float32([[1,0, shift[1]], [0,1, shift[0]]])
      rows,cols = label_eat.shape
      label_eat = cv.warpAffine(label_eat, M, (cols,rows), flags=cv.INTER_NEAREST)
      label_peri = cv.warpAffine(label_peri, M, (cols,rows), flags=cv.INTER_NEAREST)

      file_name = f'{patient}_{label_file}'
      np.save(p.join(folder, 'input', file_name.replace('.png', '.npy')), input)
      cv.imwrite(p.join(folder, 'label', file_name), label_peri)




