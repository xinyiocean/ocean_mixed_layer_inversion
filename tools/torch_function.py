#!/usr/bin/env python

# this is a function for pytorch training process
def bp_train(feature, label, model, loss_alg, optimizer):
    output = model(feature)
    loss = loss_alg(output,label)
    if loss.item()<=5: 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return output, loss.item(), model

def test(feature, label, model, loss_alg):
    output = model(feature)
    loss = loss_alg(output,label)
    return output, loss.item()
