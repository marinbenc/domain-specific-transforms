import test

datasets = ['dermis', 'dermquest', 'isic']

for dataset_train in datasets:
  for dataset_test in datasets:
    if dataset_test == dataset_train:
      continue

    print('Train on {} and test on {} ...'.format(dataset_train, dataset_test))
    stnunet_results = test.test(
      model_type='fine', 
      dataset='lesion', 
      log_name=dataset_train + '_stnunet', 
      dataset_folder='all', 
      subset=dataset_test, 
      transformed_images=False,
      save_predictions=True)
    
    stnunet_results.to_csv(f'results/stnunet_train={dataset_train}_test={dataset_test}.csv')

    unet_results = test.test(
      model_type='seg', 
      dataset='lesion', 
      log_name=dataset_train + '_baseline', 
      dataset_folder='all', 
      subset=dataset_test, 
      transformed_images=False,
      save_predictions=True)

    unet_results.to_csv(f'results/unet_train={dataset_train}_test={dataset_test}.csv')
