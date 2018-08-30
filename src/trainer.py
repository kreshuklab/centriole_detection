import torch.nn.functional as F

def train(model, train_dl, criterion, optimizer):
    model.train()
    criterion.train()

    global_loss = 0.0
    
    for inputs, label in train_dl:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        global_loss += loss.item()
    
    global_loss /= len(train_dl)
    return global_loss
    

def validate(model, test_dl, criterion):
    model.eval()
    criterion.eval()

    global_loss = 0.0
    accuracy    = 0.0 
    
    for inputs, label in test_dl:
        outputs = model(inputs)
        loss = criterion(outputs, label)
        global_loss += loss.item()
        accuracy    += (round(float(F.softmax(outputs, dim=1)[0][0])) == label)
    
    global_loss /= len(test_dl)
    accuracy    /= len(test_dl)


    return global_loss, accuracy