from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle
from xml.sax.handler import all_features


from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

DATA_PATH = r"C:\Users\priya\Documents\split_dataset_new"

import os
from glob import glob

from numpy import shape

def get_image_paths(data_path, categories):
    
    num_train_per_cat = 80  # Number of training samples per category
    num_test_per_cat = 20   # Number of test samples per category
    num_val_per_cat = 10    # Number of validation samples per category

    num_categories = len(categories)

    train_image_paths = []
    test_image_paths = []
    val_image_paths = []

    train_labels = []
    test_labels = []
    val_labels = []

    for category in categories:

        image_paths = glob(os.path.join(data_path, 'train', category, '*.jpg'))
        # print(image_paths)
        for i in range(num_train_per_cat):  
            train_image_paths.append(image_paths[i])
            #print(train_image_paths)
            train_labels.append(category)

        image_paths = glob(os.path.join(data_path, 'test', category, '*.jpg'))
        for i in range(num_test_per_cat):
            test_image_paths.append(image_paths[i])
            test_labels.append(category)
            
        image_paths = glob(os.path.join(data_path, 'val', category, '*.jpg'))
        for i in range(num_val_per_cat):
            val_image_paths.append(image_paths[i])
            val_labels.append(category)
    print("Train labels:",shape(train_labels))
    print("Test labels:", shape(test_labels))
    print("Val Labels:", shape(val_labels))

    return train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels
    # return train_image_paths, test_image_paths, train_labels, test_labels

CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
              'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['agr', 'pln', 'bbd', 'bch', 'bld', 'chp', 'drs',
                   'for', 'frw', 'gof', 'hrb', 'int', 'mrs', 'mhp',
                   'ops', 'pkb', 'riv', 'rwy', 'srs', 'stg', 'tns']

# NUM_TRAIN_PER_CAT = 70

def main():
    
    def combine_features_and_labels(train_feats, test_feats, val_feats, train_labels, test_labels, val_labels):
        """
        Combine features and labels from training, testing, and validation datasets.
        """
        # Combine all features and labels into single arrays
        all_feats = np.vstack([train_feats, test_feats, val_feats])
        all_labels = train_labels + test_labels + val_labels  # Concatenate label lists
        return all_feats, all_labels

    def visualize_tsne(features, labels, categories, perplexity=30, learning_rate=200, save_path="tsne_visualization.png"):
        """
        Visualize high-dimensional features using t-SNE and save the plot.
        
        Args:
            features (ndarray): High-dimensional feature vectors.
            labels (list): Labels corresponding to features.
            categories (list): List of category names.
            perplexity (int): Perplexity parameter for t-SNE.
            learning_rate (int): Learning rate for t-SNE.
            save_path (str): Path to save the t-SNE visualization image.
        """
        # Apply t-SNE to reduce dimensions to 2D
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
        reduced_features = tsne.fit_transform(features)

        # Map labels to numeric values for coloring
        label_to_idx = {label: idx for idx, label in enumerate(categories)}
        numeric_labels = [label_to_idx[label] for label in labels]

        # Scatter plot of t-SNE reduced features
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            c=numeric_labels,
            cmap="tab10",
            s=10,
            alpha=0.7,
        )
        plt.colorbar(scatter, ticks=range(len(categories)), label="Categories")
        plt.title("t-SNE Visualization of SIFT Features")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True)

        # Manually create category legend
        for label, color in zip(categories, plt.cm.tab10.colors):
            plt.scatter([], [], c=[color], label=label, s=50, edgecolors="k")
        plt.legend(title="Categories", loc="best")

        # Save the plot to a file
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("t-SNE visualization saved to: ", save_path)

        plt.show()
  
    def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories, output_path='confusion_matrix.png'):
        """
        Build and save the confusion matrix as an image.
        """
        # Compute confusion matrix
        cm = confusion_matrix(test_labels_ids, predicted_categories)
        np.set_printoptions(precision=2)

        # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized Confusion Matrix')
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print("Confusion matrix saved to:", output_path)
        plt.show()

    def plot_confusion_matrix(cm, categories, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        Plot and label the confusion matrix.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(categories))
        plt.xticks(tick_marks, categories, rotation=45)
        plt.yticks(tick_marks, categories)
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    def calculate_accuracy(test_labels, predicted_categories, categories, file_path='accuracy.txt'):
        """
        Calculate and display the overall and per-category accuracy.
        """
        total_accuracy = sum(1 for x, y in zip(test_labels, predicted_categories) if x == y) / len(test_labels)
        print("Overall Accuracy = {:.2f}".format(total_accuracy))

        for category in categories:
            category_correct = sum(1 for x, y in zip(test_labels, predicted_categories) if x == y and x == category)
            category_total = test_labels.count(category)
            category_accuracy = category_correct / category_total if category_total > 0 else 0
            print("{}: {:.2f}".format(category, category_accuracy))
        #save it in a text file and also the name of the file should be the confusion_matrix_classifier_vocab_size.txt
        with open(file_path, 'w') as f:
            f.write("Overall Accuracy = {:.2f}\n".format(total_accuracy))
            for category in categories:
                category_correct = sum(1 for x, y in zip(test_labels, predicted_categories) if x == y and x == category)
                category_total = test_labels.count(category)
                category_accuracy = category_correct / category_total if category_total > 0 else 0
                f.write("{}: {:.2f}\n".format(category, category_accuracy)) 

    def classify_and_evaluate(train_feats, train_labels, test_feats, test_labels, classifier_name, classifier_fn, abbr_categories, output_path):
            """
            Classify using the given classifier and evaluate performance.
            """
            print("Classifying using", classifier_name,"...")
            if classifier_name == 'nearest_neighbor':
                # YOU CODE nearest_neighbor_classify.py
                predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
                total_accuracy_knn = sum(1 for x, y in zip(test_labels, predicted_categories) if x == y) / len(test_labels)
                accuracy_knn.append(total_accuracy_knn)

            elif classifier_name == 'support_vector_machine':
                # YOU CODE svm_classify.py
                predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
                total_accuracy_svm = sum(1 for x, y in zip(test_labels, predicted_categories) if x == y) / len(test_labels)
                accuracy_svm.append(total_accuracy_svm)
            else:
                print("Invalid classifier name:", classifier_name)
                raise ValueError("Invalid classifier name:", classifier_name)
            
            file_path = "accuracy_{}_{}.txt".format(classifier_name, vocab_size)

            # Calculate accuracy
            calculate_accuracy(test_labels, predicted_categories, CATEGORIES, file_path)

            # Convert labels to numeric IDs
            test_labels_ids = [CATE2ID[label] for label in test_labels]
            predicted_categories_ids = [CATE2ID[label] for label in predicted_categories]

            # Build and save confusion matrix
            build_confusion_mtx(test_labels_ids, predicted_categories_ids, abbr_categories, output_path)

    
    def svm_multi_classify(train_image_feats, train_labels, test_image_feats):
        # klearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
        # use different kernels like rbf, linear, ploy etc 
        Linear_svc = LinearSVC(C=700.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter= 2000, multi_class='ovr', penalty='l2', random_state=0, tol= 1e-4, verbose=0)
        poly_svc = SVC(kernel='poly', degree=3, gamma='scale', coef0=0.0, C=1.0, verbose=0, max_iter=-1, decision_function_shape='ovr', random_state=None)
        rbf_svc = SVC(kernel='rbf', gamma='scale', C=1.0, verbose=0, max_iter=-1, decision_function_shape='ovr', random_state=None)
        sigmoid_svc = SVC(kernel='sigmoid', gamma='scale', coef0=0.0, C=1.0, verbose=0, max_iter=-1, decision_function_shape='ovr', random_state=None)
        # fit the model
        Linear_svc.fit(train_image_feats, train_labels)
        poly_svc.fit(train_image_feats, train_labels)
        rbf_svc.fit(train_image_feats, train_labels)
        sigmoid_svc.fit(train_image_feats, train_labels)
        # predict the model
        pred_label_linear = Linear_svc.predict(test_image_feats)
        pred_label_poly = poly_svc.predict(test_image_feats)
        pred_label_rbf = rbf_svc.predict(test_image_feats)
        pred_label_sigmoid = sigmoid_svc.predict(test_image_feats)
        
        return pred_label_linear, pred_label_poly, pred_label_rbf, pred_label_sigmoid
        
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels = \
        get_image_paths(DATA_PATH, CATEGORIES)
        
    print("Train labels:",np.shape(train_labels))
    print("Test labels:", np.shape(test_labels))

    vocab_sizes = [50, 80, 100, 120, 150, 200]
    # vocab_sizes = [200]
    accuracy_knn = []
    accuracy_svm = []
    
    for vocab_size in vocab_sizes:
        print("Vocabulary size: ", vocab_size)
        
        if os.path.isfile('vocab.pkl') is False:
            print('No existing visual word vocabulary found. Computing one from training images\n')            
            vocab = build_vocabulary(train_image_paths, vocab_size)
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.isfile('train_image_feats_1.pkl') is False:
            train_image_feats = get_bags_of_sifts(train_image_paths);
            with open('train_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats_1.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)

        if os.path.isfile('test_image_feats_1.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths);
            with open('test_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats_1.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)
        
        if os.path.isfile('val_image_feats_1.pkl') is False:
            val_image_feats  = get_bags_of_sifts(val_image_paths);
            with open('val_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(val_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('val_image_feats_1.pkl', 'rb') as handle:
                val_image_feats = pickle.load(handle)

        print("Concatenating features and labels for t-SNE visualization")
        all_feats, all_labels = combine_features_and_labels(train_image_feats, test_image_feats, val_image_feats, train_labels, test_labels, val_labels)
        print("Visualizing t-SNE")
        #visualize_tsne(all_feats, all_labels, CATEGORIES, save_path="tsne_visualization_{}.png".format(vocab_size))
        
        
        # Step 2: Classify each test image by training and using the appropriate classifier & save results
        print("Classifying and evaluating using KNN and SVM")
        

        # Classify with SVM
        classify_and_evaluate(
            train_image_feats,
            train_labels,
            test_image_feats,
            test_labels,
            "support_vector_machine",
            svm_classify,
            ABBR_CATEGORIES,
            output_path="confusion_matrix_svm_{}.png".format(vocab_size)
        )
        
        # Step 3: Classify using multi-class SVM
        print("Classifying using multi-class SVM...")
        pred_label_linear, pred_label_poly, pred_label_rbf, pred_label_sigmoid = svm_multi_classify(train_image_feats, train_labels, test_image_feats)
        
        # bar graph for multi-class SVM
        accuracy_linear_svc = []
        accuracy_poly_svc = []
        accuracy_rbf_svc = []
        accuracy_sigmoid_svc = []
        for category in CATEGORIES:
            category_correct_linear = sum(1 for x, y in zip(test_labels, pred_label_linear) if x == y and x == category)
            category_total_linear = test_labels.count(category)
            category_accuracy_linear = category_correct_linear / category_total_linear if category_total_linear > 0 else 0
            accuracy_linear_svc.append(category_accuracy_linear)
        
            
            category_correct_poly = sum(1 for x, y in zip(test_labels, pred_label_poly) if x == y and x == category)
            category_total_poly = test_labels.count(category)
            category_accuracy_poly = category_correct_poly / category_total_poly if category_total_poly > 0 else 0
            accuracy_poly_svc.append(category_accuracy_poly)
            
            category_correct_rbf = sum(1 for x, y in zip(test_labels, pred_label_rbf) if x == y and x == category)
            category_total_rbf = test_labels.count(category)
            category_accuracy_rbf = category_correct_rbf / category_total_rbf if category_total_rbf > 0 else 0
            accuracy_rbf_svc.append(category_accuracy_rbf)
            
            category_correct_sigmoid = sum(1 for x, y in zip(test_labels, pred_label_sigmoid) if x == y and x == category)
            category_total_sigmoid = test_labels.count(category)
            category_accuracy_sigmoid = category_correct_sigmoid / category_total_sigmoid if category_total_sigmoid > 0 else 0
            accuracy_sigmoid_svc.append(category_accuracy_sigmoid)

        print(accuracy_linear_svc)    
        # plot the bar graph
        fig, ax = plt.subplots()
        bar_width = 0.2
        index = np.arange(len(CATEGORIES))
        opacity = 0.8
        ax.bar(index, accuracy_linear_svc, bar_width, alpha=opacity, color='b', label='Linear SVC')
        ax.bar(index + bar_width, accuracy_poly_svc, bar_width, alpha=opacity, color='g', label='Poly SVC')
        ax.bar(index + 2*bar_width, accuracy_rbf_svc, bar_width, alpha=opacity, color='r', label='RBF SVC')
        ax.bar(index + 3*bar_width, accuracy_sigmoid_svc, bar_width, alpha=opacity, color='y', label='Sigmoid SVC')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy of Multi-class SVM')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(CATEGORIES)
        ax.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('accuracy_multi_class_svm{}.png'.format(vocab_size))
        plt.show()
        
        # removing the pkl files 
        os.remove('train_image_feats_1.pkl')
        os.remove('test_image_feats_1.pkl')
        os.remove('val_image_feats_1.pkl')
        os.remove('vocab.pkl')
        
    # Step 4: Plot accuracy vs. vocabulary size
    plt.figure()
    plt.plot(vocab_sizes, accuracy_knn, label='KNN', marker='o')
    plt.plot(vocab_sizes, accuracy_svm, label='SVM', marker='s')
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Vocabulary Size')
    plt.ylim(1, 100)  # Set y-axis range to 1-100
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Optional: Add a grid for better readability
    plt.savefig('accuracy_vs_vocab_size.png')
    plt.show()



if __name__ == '__main__':
    main()