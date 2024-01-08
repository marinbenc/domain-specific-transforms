import test

dataset_combinations = [
  ('dermis', 'dermquest'),
  ('isic', 'ph2')
]

dataset_combinations += [(d2, d1) for (d1, d2) in dataset_combinations]

for dataset_train, dataset_test in dataset_combinations:

  # Test in-sample

  print('Train on {} and test on {} ...'.format(dataset_train, dataset_train))

  stnunet_results = test.test(
    model_type='fine', 
    dataset='lesion', 
    log_name=dataset_train + '_final', 
    dataset_folder='test', 
    subset=dataset_train, 
    transformed_images=False,
    save_predictions=False)

  stnunet_results.to_csv(f'results/stnunet_train={dataset_train}_test={dataset_train}.csv')

  unet_results = test.test(
    model_type='seg', 
    dataset='lesion', 
    log_name=dataset_train + '_final', 
    dataset_folder='test', 
    subset=dataset_train, 
    transformed_images=False,
    save_predictions=False)

  unet_results.to_csv(f'results/unet_train={dataset_train}_test={dataset_train}.csv')

  # Test out-of-sample

  print('Train on {} and test on {} ...'.format(dataset_train, dataset_test))
  stnunet_results = test.test(
    model_type='fine', 
    dataset='lesion', 
    log_name=dataset_train + '_final', 
    dataset_folder='all', 
    subset=dataset_test, 
    transformed_images=False,
    save_predictions=False)
  
  stnunet_results.to_csv(f'results/stnunet_train={dataset_train}_test={dataset_test}.csv')

  unet_results = test.test(
    model_type='seg', 
    dataset='lesion', 
    log_name=dataset_train + '_final', 
    dataset_folder='all', 
    subset=dataset_test, 
    transformed_images=False,
    save_predictions=False)

  unet_results.to_csv(f'results/unet_train={dataset_train}_test={dataset_test}.csv')
