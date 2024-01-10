import test
import data.datasets as data
import pandas as pd

dataset_combinations = [
  ('dermis', 'dermquest'),
  ('isic', 'ph2')
]

dataset_combinations += [(d2, d1) for (d1, d2) in dataset_combinations]

def df_array_to_df(df_array):
  return pd.concat(df_array)

for dataset_train, dataset_test in dataset_combinations:

  # Test in-sample

  print('Train on {} and test on {} ...'.format(dataset_train, dataset_train))

  in_sample_data = data.get_kfolds_datasets('lesion', dataset_train, 5, dataset_train + '_final')
  in_sample_stnunet_results = []
  in_sample_unet_results = []

  for fold, (_, val_dataset) in enumerate(in_sample_data):
    stnunet_results = test.test(
      model_type='fine',
      log_name=dataset_train + '_final',
      fold=fold,
      test_dataset=val_dataset,
      save_predictions=False)
    
    in_sample_stnunet_results.append(stnunet_results)

    unet_results = test.test(
      model_type='seg',
      log_name=dataset_train + '_final',
      fold=fold,
      test_dataset=val_dataset,
      save_predictions=False)

    in_sample_unet_results.append(unet_results)

  in_sample_stnunet_results = df_array_to_df(in_sample_stnunet_results)
  in_sample_unet_results = df_array_to_df(in_sample_unet_results)

  in_sample_stnunet_results.to_csv(f'results/stnunet_train={dataset_train}_test={dataset_train}.csv')
  in_sample_unet_results.to_csv(f'results/unet_train={dataset_train}_test={dataset_train}.csv')
  
  # Test out-of-sample

  print('Train on {} and test on {} ...'.format(dataset_train, dataset_test))

  oos_unet_results = []
  oos_stnunet_results = []
  test_dataset = data.get_valid_dataset('lesion', dataset_test, subjects='all')

  for i in range(5):
    stnunet_results = test.test(
      model_type='fine',
      log_name=dataset_train + '_final',
      fold=i,
      test_dataset=test_dataset,
      save_predictions=False)

    oos_stnunet_results.append(stnunet_results)

    unet_results = test.test(
      model_type='seg',
      log_name=dataset_train + '_final',
      fold=i,
      test_dataset=test_dataset,
      save_predictions=False)

    oos_unet_results.append(unet_results)

  # take average across folds for each image

  oos_stnunet_result_df = oos_stnunet_results[0]
  for df in oos_stnunet_results[1:]:
    oos_stnunet_result_df = oos_stnunet_result_df + df
  oos_stnunet_result_df = oos_stnunet_result_df / len(oos_stnunet_results)

  oos_unet_result_df = oos_unet_results[0]
  for df in oos_unet_results[1:]:
    oos_unet_result_df = oos_unet_result_df + df
  oos_unet_result_df = oos_unet_result_df / len(oos_unet_results)

  oos_stnunet_result_df.to_csv(f'results/stnunet_train={dataset_train}_test={dataset_test}.csv')
  oos_unet_result_df.to_csv(f'results/unet_train={dataset_train}_test={dataset_test}.csv')