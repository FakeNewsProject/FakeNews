# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 23:47:03 2017

@author: Rui Silva

"""
from numpy import random
from random import shuffle
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import pandas as pd
import numpy as np

alpha = 20
mu = 0.1


def quality_cdf(u):
    return u

    
class Meme:
    def __init__(self, quality, views=0, shares=0, start=0):
        self.quality = quality
        self.views = views
        self.shares = shares
        self.start = start
        self.end = np.nan
        self.feeds_nb = 0
        

class Person:
    def __init__(self, feed = [], friends = []):
        self.feed = []
        self.friends = []
        
    def publish(self, network, quality, views=0, shares=0):
        mem = Meme(quality, views, shares, network.time)
        self.share(mem)
        network.memes.append(mem)
        
    def view(self, meme):
        meme.views += 1
        meme.feeds_nb += 1

        if meme in self.feed:
            self.feed.remove(meme)

        self.feed.append(meme)

        if len(self.feed) > alpha:
            meme2 = self.feed[0]
            meme2.feeds_nb += -1
            self.feed.pop(0)
            if meme2.feeds_nb == 0:
                meme2.end = 0

    def share(self, meme):
        meme.shares += 1
        for friend in self.friends:
            friend.view(meme)
            
    def read_feed(self, u_sample):
        quality_sum = 0
        temp = 0
        
        for meme in self.feed:
            quality_sum += meme.quality
        
        for meme in self.feed:
            temp += meme.quality / quality_sum
            if u_sample <= temp: 
                self.share(meme)
                break
        
    def action(self, network):
        u_sample = random.uniform(0, 1)
        if u_sample >= 1-mu:
            self.publish(network=network, quality=quality_cdf((1 - u_sample) / mu))
        else:
            self.read_feed(u_sample / (1 - mu))
    
    def connected(self, person):
        if person == self: return True
        else: return (person in self.friends) and (self in person.friends)
            
    def connect(self, person):
        if not self.connected(person):
            self.friends.append(person)
            person.connect(self)
            

class Network:
    def __init__(self, people=0, connexions=0):
        self.time = 0
        self.people = []
        self.memes = []
        self.size = 0
        if connexions > people * (people - 1):
            raise Exception('Not enough people in the network to create '+str(connexions)+' different connexions')
        
        for i in range(people): self.add_person()
        self.random_connexions(connexions)
            
    def add_person(self):
        self.people.append(Person(feed=[], friends=[]))
        self.size += 1
        
    def connected(self, person1, person2):
        person1.connected(person2)
        
    def connect(self, person1, person2):
        person1.connect(person2)
        
    def random_connexions(self, n):
        connexions = [(i, j) for i in range(self.size) for j in range(self.size)]
        shuffle(connexions)
        count = 0
        temp = 0
        while count < n:
            (k, l) = connexions[count + temp]
            person1 = self.people[k]
            person2 = self.people[l]
            if self.connected(person1, person2):
                temp += 1
            else: 
                self.connect(person1, person2)
                count += 1

    def timestep(self):
        self.time += 1
        if self.time % 1000 == 0: print(self.time, sep=',')
        temp = random.randint(self.size)
        person = self.people[temp]
        person.action(self)

                
    def simulate(self, n_steps):
        temp = random.randint(self.size, size=n_steps)
        for i in range(n_steps):
            if i % 200 == 0 : print(i),
            person = self.people[temp[i]]
            person.action(self, i)
            
    def plot_memes(self):
        x = [meme.quality for meme in self.memes]
        y = [meme.shares for meme in self.memes]
        plt.scatter(x, y)
        plt.xlabel("Quality")
        plt.ylabel("Shares")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()
        
    def kendall_tau(self):
        x = [meme.quality for meme in self.memes]
        y = [meme.shares for meme in self.memes]
        (tau, p) = stats.kendalltau(x, y)
        return tau


if __name__ == '__main__':

    n_steps = 10000
    n_people = 1000
    n_connexions = 100000

    t = time.time()

    net = Network(people=n_people, connexions=n_connexions)
    for i in range(n_steps):
        net.timestep()
    net.plot_memes()
    
    memes_df = pd.DataFrame({'quality': [meme.quality for meme in net.memes],
                             'views': [meme.views for meme in net.memes],
                             'shares': [meme.shares for meme in net.memes]})
    
    print(memes_df)
    
    print("Kendall Tau : " + str(net.kendall_tau()))
            
    print("Exec time : "+str(time.time() - t)+" seconds")   
