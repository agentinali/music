# Load the DenseNet
model_conv = torchvision.models.densenet201(pretrained=True)

# Freeze all layers in the network
for param in model_conv.parameters():
    param.requires_grad = False

# Get the number of inputs of the last layer (or number of neurons in the layer preceeding the last layer)
num_ftrs = model_conv.classifier.in_features

# Reconstruct the last layer (output layer) to have only two classes
model_conv.classifier = nn.Linear(num_ftrs, 2)

# Initiate the model on GPU
if torch.cuda.is_available():
    model_conv = model_conv.cuda()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.parameters(), lr=0.001,
                      momentum=0.9)  # Try Adam optimizer for better accuracy: optim.Adam(model_conv.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# Training
import time

num_epochs = 20
for epoch in range(num_epochs):
    start = time.time()
    exp_lr_scheduler.step()
    # Reset the correct to 0 after passing through all the dataset
    correct = 0
    for images, labels in dataloaders['training_set']:
        images = Variable(images)
        labels = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model_conv(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()

    train_acc = 100 * correct / dataset_sizes['training_set']
    stop = time.time()
    print('Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {}%, Time: {:.2f}s'
          .format(epoch + 1, num_epochs, loss.item(), train_acc, stop - start))

    # Testing
model_conv.eval()
with torch.no_grad():
    correct = 0
    total = 0
    start = time.time()
    for (images, labels) in dataloaders['test_set']:

        images = Variable(images)
        labels = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = model_conv(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    stop = time.time()

    print('Test Accuracy: {:.3f} %, Time: {:.2f}s'.format(100 * correct / total, stop - start))

# Import your trial images and check against the model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Predict your own image
def predict(img_name, model):
    image = cv2.imread(img_name)  # Read the image
    # ret, thresholded = cv2.threshold(image,127,255,cv2.THRESH_BINARY)   #Threshold the image
    img = Image.fromarray(image)  # Convert the image to an array
    img = transforms_photo(img)  # Apply the transformations
    img = img.view(1, 3, 224, 224)  # Add batch size
    img = Variable(img)
    # Wrap the tensor to a variable

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    output = model(img)
    print(output)
    print(output.data)
    _, predicted = torch.max(output, 1)
    if predicted.item() == 0:
        p = 'Cat'
    else:
        p = 'Dog'
    cv2.imshow('Original', image)
    return p