import os
from glob import glob
from PIL import Image
def get_image_paths(data_path, categories, num_train_per_cat, num_test_per_cat):
    num_categories = len(categories)

    train_image_paths = []
    test_image_paths = []

    train_labels = []
    test_labels = []

    for category in categories:

        image_paths = glob(os.path.join(data_path, 'train', category, '*.tif'))
        for i in range(num_train_per_cat):
            temp=image_paths[i]
            fin= temp.replace('split_dataset','split_dataset_new')
            if not os.path.isdir(os.path.dirname(fin)):
                os.makedirs(os.path.dirname(fin))
            Image.open(temp).convert('L').resize((72,72)).save(fin)
            train_image_paths.append(fin)
            train_labels.append(category)

        image_paths = glob(os.path.join(data_path, 'test', category, '*.tif'))
        for i in range(num_test_per_cat):
            temp=image_paths[i]
            fin= temp.replace('split_dataset','split_dataset_new')
            if not os.path.isdir(os.path.dirname(fin)):
                os.makedirs(os.path.dirname(fin))
            Image.open(temp).convert('L').resize((72,72)).save(fin)
            test_image_paths.append(fin)
            #test_image_paths.append(temp)
            test_labels.append(category)


    return train_image_paths, test_image_paths, train_labels, test_labels
