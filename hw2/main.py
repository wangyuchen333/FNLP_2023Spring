import argparse
from utils import get_features,get_shuffle
from model import LogLinearModel

parser = argparse.ArgumentParser(description='Log-Linear model for text classification')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--n_classes',default=20, type=int, help='number of classes')
parser.add_argument('--n_features',default=20000, type=int, help='number of the demension of features')
parser.add_argument('--batch_size',default=10000,type=int, help='batch size')
parser.add_argument('--epochs',default=18,type=int, help='number of training epochs')
parser.add_argument('--lemma',default = 0.1,type=float, help='regularization coefficient')
args = parser.parse_args()

training_accs = []
test_accs = []
log_file = r'record.txt'

'''
Extract features and labels for the csv files
'''
training_features,training_labels,testing_features,testing_labels,word2id = get_features("train.csv","test.csv",dimension=args.n_features)

'''
Build the model
'''
model = LogLinearModel(lr=args.lr,n_classes=args.n_classes,n_features=args.n_features,lemma=args.lemma)


'''
Train and evaluate
'''
# best_acc = 0
print("start training")
for i in range(args.epochs):
    get_shuffle(training_features,training_labels)
    model.train(training_features,training_labels,batch_size=args.batch_size)
    test_acc = model.test(testing_features,testing_labels) 
    train_acc = model.test(training_features,training_labels)
    training_accs.append(train_acc)
    test_accs.append(test_acc)
    # best_acc = max(test_acc,best_acc)

test_acc = model.test(testing_features, testing_labels)
train_acc = model.test(training_features, training_labels)
sktest_acc = model.sktest(testing_features, testing_labels)
sktrain_acc = model.sktest(training_features, training_labels)
print("training accuracy = {},test accuracy = {},"
        "sktraining accuracy = {},sktest accuracy = {}".format(train_acc,test_acc,sktrain_acc, sktest_acc))
