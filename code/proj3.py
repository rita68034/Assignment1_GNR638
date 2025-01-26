from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle
import sys
from sklearn.model_selection import train_test_split, KFold

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

# Step 0: Set up parameters, category list, and image paths.

#For this project, you will need to report performance for three
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier
# 3) Bag of sift features and linear SVM classifier
#The starter code is initialized to 'placeholder' just so that the starter
#code does not crash when run unmodified and you can get a preview of how
#results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()

#DATA_PATH = '../data/'
#DATA_PATH = r'C:\Users\priya\Downloads\UCMerced_LandUse\UCMerced_LandUse\Images'
DATA_PATH = r"C:\Users\priya\Documents\split_dataset"
#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

# CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
#              'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
#               'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']
CATEGORIES = ['agricultural',
'airplane',
'baseballdiamond',
'beach',
'buildings',
'chaparral',
'denseresidential',
'forest',
'freeway',
'golfcourse',
'harbor',
'intersection',
'mediumresidential',
'mobilehomepark',
'overpass',
'parkinglot',
'river',
'runway',
'sparseresidential',
'storagetanks',
'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For', 'a','b','c','d','e','f']


FEATURE = args.feature
# FEATUR  = 'bag of sift'

CLASSIFIER = args.classifier
# CLASSIFIER = 'support vector machine'

#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as
#well.

NUM_TRAIN_PER_CAT = 70
NUM_TEST_PER_CAT = 20


def main():
    #This function returns arrays containing the file path for each train
    #and test image, as well as arrays with the label of each train and
    #test image. By default all four of these arrays will be 1500 where each
    #entry is a string.
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT, NUM_TEST_PER_CAT)
    #print(train_image_paths[:10])
    #sys.exit(0)
    
    codewords= [50] # 75, 100, 150, 200, 250]
    final_accuracy = []
    for vocab_size in codewords:
        kf = KFold(n_splits=8)
        accuracies=[]
        for train_index, val_index in kf.split(train_image_paths):
            print("priya")
            train_image_paths_new, val_image_paths_new = np.array(train_image_paths)[train_index].tolist(), np.array(train_image_paths)[val_index].tolist()
            train_labels_new, val_labels_new = np.array(train_labels)[train_index].tolist(), np.array(train_labels)[val_index].tolist()
            train_image_feats, test_image_feats= creating_features(train_image_paths_new, val_image_paths_new, vocab_size)
            predicted_categories = svm_classify(train_image_feats, train_labels_new, test_image_feats)
            accuracy = float(len([x for x in zip(val_labels_new,predicted_categories) if x[0]== x[1]]))/float(len(val_labels_new))
            print(accuracy)
            accuracies.append(accuracy)
        final_accuracy.append(sum(accuracies)/len(accuracies))
    print(final_accuracy)
    sys.exit(0)
    
    
    for category in CATEGORIES:
        accuracy_each = float(len([x for x in zip(test_labels,predicted_categories) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
        print(str(category) + ': ' + str(accuracy_each))
    
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    train_labels_ids = [CATE2ID[x] for x in train_labels]
    
   
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
    visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, train_labels_ids)

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, CATEGORIES)
    '''
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('Normalized confusion matrix')
    #print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')

    plt.show()
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def creating_features(train_image_paths, test_image_paths, vocab_size):
    vocab = build_vocabulary(train_image_paths, vocab_size)
    with open('vocab.pkl', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    train_image_feats = get_bags_of_sifts(train_image_paths)
    with open('train_image_feats_1.pkl', 'wb') as handle:
        pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    test_image_feats  = get_bags_of_sifts(test_image_paths)
    with open('test_image_feats_1.pkl', 'wb') as handle:
        pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return train_image_feats, test_image_feats



if __name__ == '__main__':
    main()

#code 2
# import os
# import argparse
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# from random import shuffle
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import confusion_matrix
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE


# from get_image_paths import get_image_paths
# from get_tiny_images import get_tiny_images
# from build_vocabulary import build_vocabulary
# from get_bags_of_sifts import get_bags_of_sifts
# from nearest_neighbor_classify import nearest_neighbor_classify
# from svm_classify import svm_classify
# from visualize import visualize

# # Function to calculate classification accuracy
# def calculate_accuracy(true_labels, predicted_labels):
#     return np.mean(np.array(true_labels) == np.array(predicted_labels))

# # Function to plot confusion matrix
# def plot_confusion_matrix(cm, category_names, title='Confusion Matrix', cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(category_names))
#     plt.xticks(tick_marks, category_names, rotation=45)
#     plt.yticks(tick_marks, category_names)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()

# def main():
#     # Argument parsing
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--feature', help='Feature type', type=str, default='bag_of_sift')
#     parser.add_argument('--classifier', help='Classifier type', type=str, default='nearest_neighbor')
#     args = parser.parse_args()

#     DATA_PATH = r"C:\Users\priya\Documents\split_dataset"
#     CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 
#                   'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 
#                   'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass',
#                   'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']

#     NUM_TRAIN_PER_CAT = 70
#     NUM_TEST_PER_CAT = 20
#     FEATURE = args.feature
#     CLASSIFIER = args.classifier

#     # Step 1: Get file paths and labels for train/test images
#     train_image_paths, test_image_paths, train_labels, test_labels = \
#         get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT, NUM_TEST_PER_CAT)
#     kf = KFold(n_splits=8)
#     for train_index, val_index in kf.split(train_image_paths):
#         print(len(train_index))
#         print(len(val_index))
#     sys.exit(0)
#     test_image_paths

#     # Step 2: Feature extraction based on selected feature type
    
     
#     if os.path.isfile('vocab.pkl') is False:
#         vocab_size = 50  # Change vocab size as needed
#         vocab = build_vocabulary(train_image_paths, vocab_size)
#         with open('vocab.pkl', 'wb') as handle:
#             pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     if os.path.isfile('train_image_feats.pkl') is False:
#         train_image_feats = get_bags_of_sifts(train_image_paths)
#         with open('train_image_feats.pkl', 'wb') as handle:
#             pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     if os.path.isfile('test_image_feats.pkl') is False:
#         test_image_feats = get_bags_of_sifts(test_image_paths)
#         with open('test_image_feats.pkl', 'wb') as handle:
#             pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     # Step 3: K-fold cross-validation
#     kf = KFold(n_splits=5)  # 5-fold cross-validation
#     accuracies = []

#     for train_index, val_index in kf.split(train_image_feats):
#         X_train, X_val = np.array(train_image_feats)[train_index], np.array(train_image_feats)[val_index]
#         y_train, y_val = np.array(train_labels)[train_index], np.array(train_labels)[val_index]

#             predicted_categories = svm_classify(X_train, y_train, X_val)

#         # Calculate accuracy for this fold
#         accuracy = calculate_accuracy(y_val, predicted_categories)
#         accuracies.append(accuracy)

#     # Step 4: Print average accuracy and confusion matrix
#     print("Average Accuracy = ", np.mean(accuracies))

#     # Step 5: Visualize confusion matrix on the full test set
#     if CLASSIFIER == 'nearest_neighbor':
#         predicted_categories_full = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
#     elif CLASSIFIER == 'support_vector_machine':
#         predicted_categories_full = svm_classify(train_image_feats, train_labels, test_image_feats)
    
#     cm = confusion_matrix(test_labels, predicted_categories_full)
#     plot_confusion_matrix(cm, CATEGORIES)
#     plt.show()

#     # Step 6: t-SNE visualization of the SIFT features (dimensionality reduction)
#     pca = PCA(n_components=50)  # Reduce to 50 dimensions for t-SNE
#     pca_feats = pca.fit_transform(train_image_feats)
#     tsne = TSNE(n_components=2)
#     tsne_feats = tsne.fit_transform(pca_feats)

#     # Step 7: Visualize t-SNE features
#     plt.figure(figsize=(8, 8))
#     label_encoder = LabelEncoder()
#     encoded_labels = label_encoder.fit_transform(train_labels)

# # Modify the scatter plot code to use encoded_labels
#     scatter = plt.scatter(tsne_feats[:, 0], tsne_feats[:, 1], c=encoded_labels, cmap='jet')
#     plt.legend(handles=scatter.legend_elements()[0], labels=CATEGORIES)
#     plt.title('t-SNE Visualization of Keypoints')
#     plt.show()
# if __name__ == '__main__':
#     main()


