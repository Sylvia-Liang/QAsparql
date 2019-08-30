from lcquad_test import Orchestrator
from parser.lc_quad import LC_QaudParser
from learning.classifier.svmclassifier import SVMClassifier
from parser.qald import Qald
from parser.lc_quad_linked import LC_Qaud_Linked
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sys
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = LC_QaudParser()
    classifier1 = SVMClassifier('./output/question_type_classifier/svm.model')
    classifier2 = SVMClassifier('./output/double_relation_classifier/svm.model')
    query_builder = Orchestrator(None, classifier1, classifier2, parser, None, auto_train=True)

    print("train_question_classifier")

    scores = query_builder.train_question_classifier(file_path="./data/LC-QUAD/data.json", test_size=0.8)
    print(scores)
    y_pred = query_builder.question_classifier.predict(query_builder.X_test)
    print(accuracy_score(query_builder.y_test, y_pred))
    print(classification_report(query_builder.y_test, y_pred, digits=3))

    ds = LC_Qaud_Linked(path="./data/LC-QUAD/linked_test.json")
    ds.load()
    ds.parse()

    lcquad = []
    lc_y = []
    for qapair in ds.qapairs:
        lcquad.append(qapair.question.text)
        if "COUNT(" in qapair.sparql.query:
            lc_y.append(2)
        elif "ASK" in qapair.sparql.query:
            lc_y.append(1)
        else:
            lc_y.append(0)

    lc_y = np.array(lc_y)
    print('LIST: ', sum(lc_y==0))
    print('ASK: ', sum(lc_y == 1))
    print('COUNT: ', sum(lc_y == 2))
    np.savetxt('lcquad_question_type.csv', lc_y, delimiter=',')

    lc_pred = query_builder.question_classifier.predict(lcquad)
    print('LC-QUAD question_classifier')
    print(accuracy_score(lc_y, lc_pred))
    print(classification_report(lc_y, lc_pred, digits=4))

    classes = ['List', 'Count', 'Boolean']
    cm = confusion_matrix(lc_y, lc_pred)
    print('Before Normalization')
    print(cm)

    print('Accuracy by class: ')
    c_acc = cm.diagonal() / cm.sum(axis=1)
    print(c_acc)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('After Normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('confusion_matrix_lcquad.png')

    q_ds = Qald(Qald.qald_7)
    q_ds.load()
    q_ds.parse()

    qald = []
    q_y = []
    for qapair in q_ds.qapairs:
        qald.append(qapair.question.text)
        if "COUNT(" in qapair.sparql.query:
            q_y.append(2)
        elif "ASK" in qapair.sparql.query:
            q_y.append(1)
            x = ascii(qapair.sparql.query.replace('\n', ' ').replace('\t', ' '))
            print(x)
        else:
            q_y.append(0)

    q_y = np.array(q_y)
    print('LIST: ', sum(q_y==0))
    print('ASK: ', sum(q_y == 1))
    print('COUNT: ', sum(q_y == 2))
    np.savetxt('qald_question_type.csv', q_y, delimiter=',')

    q_pred = query_builder.question_classifier.predict(qald)
    print('QALD question_classifier')
    print(accuracy_score(q_y, q_pred))
    print(classification_report(q_y, q_pred, digits=4))

    classes = ['List', 'Count', 'Boolean']
    cm = confusion_matrix(q_y, q_pred)
    print('Before Normalization')
    print(cm)

    print('Accuracy by class: ')
    c_acc = cm.diagonal() / cm.sum(axis=1)
    print(c_acc)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('After Normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('confusion_matrix_qald.png')

    ds = LC_Qaud_Linked(path="./data/LC-QUAD/linked_answer.json")
    ds.load()
    ds.parse()

    lcquad = []
    lc_y = []
    for qapair in ds.qapairs:
        lcquad.append(qapair.question.text)
        if "COUNT(" in qapair.sparql.query:
            lc_y.append(2)
        elif "ASK" in qapair.sparql.query:
            lc_y.append(1)
        else:
            lc_y.append(0)

    lc_y = np.array(lc_y)
    print('LIST: ', sum(lc_y==0))
    print('ASK: ', sum(lc_y == 1))
    print('COUNT: ', sum(lc_y == 2))
    np.savetxt('lcquad_question_type_all.csv', lc_y, delimiter=',')

    lc_pred = query_builder.question_classifier.predict(lcquad)
    print('LC-QUAD question_classifier')
    print(accuracy_score(lc_y, lc_pred))
    print(classification_report(lc_y, lc_pred, digits=4))

    classes = ['List', 'Count', 'Boolean']
    cm = confusion_matrix(lc_y, lc_pred)
    print('Before Normalization')
    print(cm)

    print('Accuracy by class: ')
    c_acc = cm.diagonal() / cm.sum(axis=1)
    print(c_acc)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('After Normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('confusion_matrix_lcquad_all.png')
