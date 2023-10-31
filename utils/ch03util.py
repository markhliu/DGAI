import torch

def onehot_encoder(position,depth):
    onehot=torch.zeros((depth,))
    onehot[position]=1
    return onehot


def int_to_onehots(number):
    # calculate the quotient
    quotient=number//5
    # calculate the remainder
    remainder=number%5
    # convert to onehots
    onehot_quotient=onehot_encoder(quotient,20)
    onehot_remainder=onehot_encoder(remainder,5)
    # concatenate
    combined=torch.cat([onehot_quotient,onehot_remainder])
    return combined





def onehots_to_int(onehots):
    # extract quotient and remainder
    onehot_quotient=onehots[:20]
    onehot_remainder=onehots[-5:]    
    quotient=torch.argmax(onehot_quotient)
    remainder=torch.argmax(onehot_remainder)
    # concatenate
    number=5*quotient+remainder
    return number.item()



import random

def gen_sequence():
    indices = random.sample(range(10), 10)
    values = torch.tensor(indices)*5
    return values  

import numpy as np

def gen_batch():
    sequence=gen_sequence()    #A
    batch=[int_to_onehots(i).numpy() for i in sequence]    #B
    batch=np.array(batch)
    return torch.tensor(batch)


def data_to_num(data):
    multiple=torch.argmax(data[:,:20],dim=-1)
    remainder=torch.argmax(data[:,20:],dim=-1)
    num=multiple*5+remainder
    return num


import torch.nn as nn

# determine the device automatically
device="cuda" if torch.cuda.is_available() else "cpu"
# the discriminator D is a binary classifier



real_labels=torch.ones((10,1)).to(device)
fake_labels=torch.zeros((10,1)).to(device)


def train_D_G(D,G,loss_fn,optimD,optimG):
    # Generate examples of real data
    true_data=gen_batch().to(device)
    # use 1 as labels since they are real
    preds=D(true_data)
    loss_D1=loss_fn(preds,real_labels)
    optimD.zero_grad()
    loss_D1.backward()
    optimD.step()
    # train D on fake data
    noise=torch.randn(10,100).to(device)
    generated_data=G(noise)
    # use 0 as labels since they are fake
    preds=D(generated_data)
    loss_D2=loss_fn(preds,fake_labels)
    optimD.zero_grad()
    loss_D2.backward()
    optimD.step()
    
    # train G 
    noise=torch.randn(10,100).to(device)
    generated_data=G(noise)
    # use 1 as labels since G wants to fool D
    preds=D(generated_data)
    loss_G=loss_fn(preds,real_labels)
    optimG.zero_grad()
    loss_G.backward()
    optimG.step()
    return generated_data       















