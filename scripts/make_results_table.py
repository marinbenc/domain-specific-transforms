import pandas as pd
import os
from scipy.stats import ranksums
import scipy.stats as st
import numpy as np

results_folder = 'results'
result_files = os.listdir(results_folder)

datasets = ['dermis', 'isic', 'dermquest']

for dataset_train in datasets:
  dataset_result_files = [f for f in result_files if f'train={dataset_train}' in f]
  dataset_result_files.sort()

  print('train=', dataset_train)

  for dataset_test in datasets:
    dataset_test_files = [f for f in dataset_result_files if f'test={dataset_test}' in f]
    dataset_test_files.sort(reverse=True)

    dfs = []

    for f in dataset_test_files:
      result_df = pd.read_csv(os.path.join(results_folder, f))
      dfs.append(result_df)

      model = f.split('_')[0]
      if model == 'stnunet':
        model = 'STN+U-Net'
      else:
        model = 'U-Net'
      f = f.replace('.csv', '')
      train_dataset = f.split('_')[1].split('=')[-1]
      test_dataset = f.split('_')[2].split('=')[-1]

      if train_dataset == 'dermis':
        train_dataset = 'DermIS'
      elif train_dataset == 'isic':
        train_dataset = 'ISIC'
      elif train_dataset == 'dermquest':
        train_dataset = 'DermQuest'

      if test_dataset == 'dermis':
        test_dataset = 'DermIS'
      elif test_dataset == 'isic':
        test_dataset = 'ISIC'
      elif test_dataset == 'dermquest':
        test_dataset = 'DermQuest'


      def calculate_ci(series):
        ci = st.t.interval(0.95, len(series)-1, loc=np.mean(series), scale=st.sem(series))
        return ci[1] - ci[0]
      
      dsc = f'${result_df["dsc"].mean() * 100:.2f} \\pm {result_df["dsc"].std() * 100:.2f}$'
      prec = f'${result_df["prec"].mean() * 100:.2f} \\pm {result_df["prec"].std() * 100:.2f}$'
      rec = f'${result_df["rec"].mean() * 100:.2f} \\pm {result_df["rec"].std() * 100:.2f}$'
      dsc_ci = f'${calculate_ci(result_df["dsc"]) * 100:.2f}$'
      prec_ci = f'${calculate_ci(result_df["prec"]) * 100:.2f}$'
      rec_ci = f'${calculate_ci(result_df["rec"]) * 100:.2f}$'

      print(f'{test_dataset} & {model} & {dsc} ({dsc_ci}) & {prec} ({prec_ci}) & {rec} ({rec_ci}) \\\\')


    if len(dfs) > 1:
      # print('Wilcoxon signed-rank test')
      # print('dsc', ranksums(dfs[0]['dsc'], dfs[1]['dsc']))
      # print('prec', ranksums(dfs[0]['prec'], dfs[1]['prec']))
      # print('rec', ranksums(dfs[0]['rec'], dfs[1]['rec']))
      # print('')

      print('Confidence interval')
      print('unet', st.t.interval(0.95, len(dfs[0]['dsc'])-1, loc=np.mean(dfs[0]['dsc']), scale=st.sem(dfs[0]['dsc'])))
      print('')

      print('Paired t-test')
      print('dsc', st.ttest_rel(dfs[0]['dsc'], dfs[1]['dsc']))
      print('prec', st.ttest_rel(dfs[0]['prec'], dfs[1]['prec']))
      print('rec', st.ttest_rel(dfs[0]['rec'], dfs[1]['rec']))
      print('')