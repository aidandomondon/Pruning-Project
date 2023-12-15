import torchvision;
import torch.utils.data.dataloader
import torch.nn;
from tqdm import tqdm;
import os;
import numpy as np;
import copy;
import re;

# Configurations
TRAINSET_DIR = './train';
TESTSET_DIR = './test';
BATCH_SIZE = 30;
MODEL_SAVES_DIR = './saves';
EPOCHS = 80;

# Load the dataset
trainset = torchvision.datasets.FashionMNIST(TRAINSET_DIR, train=True, 
                                                transform=torchvision.transforms.ToTensor(), 
                                                download=True);
testset = torchvision.datasets.FashionMNIST(TESTSET_DIR, train=False, 
                                                transform=torchvision.transforms.ToTensor(), 
                                                download=True);
# dataloaders leverage multiprocessing to fetch data efficiently
trainloader = torch.utils.data.dataloader.DataLoader(testset, BATCH_SIZE);
testloader = torch.utils.data.dataloader.DataLoader(testset, BATCH_SIZE);

# neural network architecture we will use to predict labels
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__();

        # Hidden Layer 1: 32 Filters
        self.conv1 = torch.nn.Conv2d(
            in_channels = 1, out_channels = 32, kernel_size = (3, 3));
        self.reLU1 = torch.nn.ReLU();
        self.maxpool1 = torch.nn.MaxPool2d((2, 2));

        # Hidden Layer 2: 64 Filters
        self.conv2 = torch.nn.Conv2d(
            in_channels = 32, out_channels = 64, kernel_size = (3, 3));
        self.reLU2 = torch.nn.ReLU();
        self.maxpool2 = torch.nn.MaxPool2d((2, 2));

        # Hidden Layer 3: 64 Filters
        self.conv3 = torch.nn.Conv2d(
            in_channels = 64, out_channels = 64, kernel_size = (3, 3));
    
        # Flatten to 1D vector
        self.flatten1 = torch.nn.Flatten(1);
    
        # Linear layers
        self.linear1 = torch.nn.Linear(576, 250);
        self.linear1Activation = torch.nn.ReLU();
        self.linear2 = torch.nn.Linear(250, 125);
        self.linear2Activation = torch.nn.ReLU();
        self.linear3 = torch.nn.Linear(125, 60);
        self.linear3Activation = torch.nn.ReLU();
        self.linear4 = torch.nn.Linear(60, 10);
        self.linear4Activation = torch.nn.ReLU();
        self.linear4Clip = torch.nn.Softmax();

    def forward(self, X):
        X = self.conv1(X);
        X = self.reLU1(X);
        X = self.maxpool1(X);
        X = self.conv2(X);
        X = self.reLU2(X);
        X = self.maxpool2(X);
        X = self.conv3(X);
        X = self.flatten1(X);
        X = self.linear1(X);
        X = self.linear1Activation(X);
        X = self.linear2(X);
        X = self.linear2Activation(X);
        X = self.linear3(X);
        X = self.linear3Activation(X);
        X = self.linear4(X);
        X = self.linear4Activation(X);
        X = self.linear4Clip(X);
        return X;


if os.listdir(MODEL_SAVES_DIR) == []:

    net = Net();
    net.train();

    loss_function = torch.nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(net.parameters());

    for epoch in range(EPOCHS):
        for i, data in tqdm(enumerate(trainloader)):
            optimizer.zero_grad();

            # get predictions
            inputs, labels = data;
            logits = net(inputs);

            # compute loss, compute gradients, update gradients accordingly
            loss = loss_function(logits, labels);
            loss.backward();
            optimizer.step();

    # save/load model mechanism to avoid re-running while debugging later parts
    torch.save(net, f'{MODEL_SAVES_DIR}/last_save.pt');
else:
    net = torch.load(f'{MODEL_SAVES_DIR}/last_save.pt');

# Evaluating the model's performance on test data
net.train(mode = False);
correct = 0;
incorrect = 0;
for i, data in tqdm(enumerate(testloader)):
    inputs, labels = data;
    logits = net(inputs);
    predictions = torch.argmax(logits, 1);
    comparisons = torch.eq(predictions, labels);
    num_correct = comparisons.long().count_nonzero();
    correct += num_correct;
    incorrect += BATCH_SIZE - num_correct;

print(f'Accuracy: {correct / (correct + incorrect)}');

###########################################
# Pruning
###########################################

# Sort sensitivites
considered_layers = filter(lambda x : re.search("^conv", x), net._modules.keys())
weights_as_list = []
layer_names = [layerName for layerName in considered_layers];
for l in range(len(layer_names)):
    layer_filters = getattr(net, layer_names[l]);
    for f in range(layer_filters.weight.shape[0]):
        filter = layer_filters.weight[f];
        for k in range(filter.shape[0]):
            kernel = filter[k];
            for i, j in np.ndindex(kernel.shape):
                weights_as_list.append({'index' : (l, f, k, i, j), 'weight': kernel[i, j]});
sorted_weights = sorted(weights_as_list, key=lambda item : item['weight']);

# Prune bottom 0.01% of weights
pruning_mass = sum([torch.numel(getattr(net, layer).weight) for layer in layer_names]) // 10000

net.requires_grad_(False);

pruned_net = copy.deepcopy(net);
for idx in range(pruning_mass):
    weight = sorted_weights[idx];
    getattr(getattr(pruned_net, layer_names[weight['index'][0]]), 'weight') \
        [weight['index'][1], weight['index'][2], weight['index'][3], weight['index'][4]] = torch.Tensor([0]);

# Evaluating pruned model's performance on test data
pruned_net.train(mode = False);
new_correct = 0;
new_incorrect = 0;
print('Evaluating new performance...')
for i, data in tqdm(enumerate(testloader)):
    inputs, labels = data;
    logits = pruned_net(inputs);
    predictions = torch.argmax(logits, 1);
    comparisons = torch.eq(predictions, labels);
    num_correct = comparisons.long().count_nonzero();
    new_correct += num_correct;
    new_incorrect += BATCH_SIZE - num_correct;

print(f'Accuracy: {new_correct / (new_correct + new_incorrect)}');