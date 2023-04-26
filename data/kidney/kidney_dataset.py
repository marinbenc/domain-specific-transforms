import data.scan_dataset as scan_dataset
import numpy as np
import os.path as p
import pandas as pd

def KidneyDatasetMale(subset, pretraining, **kwargs):
  ds = KidneyDataset(subset='all', pretraining=pretraining, **kwargs)
  
  male_subjects = np.array(ds.get_subjects_with_sex('male'))
  np.random.seed(2022)
  np.random.shuffle(male_subjects)
  
  train_subjects = list(male_subjects[:int(len(male_subjects) * 0.9)])
  train_subjects.sort()
  valid_subjects = list(male_subjects[int(len(male_subjects) * 0.9):])
  valid_subjects.sort()
  test_subjects = ds.get_subjects_with_sex('female')
  test_subjects.sort()

  if subset == 'train':
    return KidneyDataset('all', pretraining, subjects=train_subjects)
  elif subset == 'valid':
    return KidneyDataset('all', pretraining, subjects=valid_subjects)
  elif subset == 'test':
    return KidneyDataset('all', pretraining, subjects=test_subjects)

class KidneyDataset(scan_dataset.ScanDataset):
  def __init__(self, subset, pretraining, **kwargs):
    super().__init__(
      subset=subset,
      pretraining=pretraining,
      dataset_folder='kidney',
      window_max=800,
      window_min=-180,
      global_min=-180,
      global_max=800,
      size=186,
      padding=8,
      stn_zoom_out=1.1,
      **kwargs
    )

    self.subjects_df = pd.read_json(p.join(p.dirname(__file__), 'data', 'kits.json'))

  def get_subjects_with_sex(self, sex):
    """
    Returns a list of subjects with the specified sex.
    """
    return self.subjects_df[self.subjects_df['gender'] == sex]['case_id'].tolist()


