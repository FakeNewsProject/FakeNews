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


def quality_cdf(u):
    return u

    
class Meme:
    def __init__(self, quality, views=0, shares=0, start=0):
        self.quality = quality
        self.views = views
        self.shares = shares
        self.ocurrences = 0
        self.start = start
        self.end = np.nan
        

class Person:
    def __init__(self, feed=[], friends=[], alpha=20, mu=0.5):
        self.alpha = alpha
        self.mu = mu
        self.feed = []
        self.friends = []
        
    def publish(self, network, quality, time, views=0, shares=0):
        mem = Meme(quality, views, shares, start=time)
        li_dead_meme = self.share(mem, time)
        network.memes.append(mem)
        network.active_memes.append(mem)
        return li_dead_meme

    def view(self, meme, time):
        meme.views += 1
        if meme in self.feed:
            self.feed.remove(meme)
            meme.ocurrences -= 1
        self.feed.append(meme)
        meme.ocurrences += 1
        if len(self.feed) > self.alpha:
            old_meme = self.feed.pop(0)
            old_meme.ocurrences -= 1
            if old_meme.ocurrences == 0:
                old_meme.end = time
                return old_meme

        
    def share(self, meme, time):
        meme.shares += 1
        li_dead_meme = []
        for friend in self.friends:
            dead_meme = friend.view(meme, time)
            if dead_meme: li_dead_meme.append(dead_meme)
        return li_dead_meme
            
    def read_feed(self, u_sample, time):
        quality_sum = 0
        temp = 0
        for meme in self.feed: quality_sum += meme.quality
        for meme in self.feed:
            temp += meme.quality / quality_sum
            if u_sample <= temp: 
                return self.share(meme, time)
        
    def action(self, network):
        u_sample = random.uniform(0, 1)
        time = network.time
        if u_sample >= 1-self.mu:
            li_dead_meme = self.publish(network=network, quality=quality_cdf((1 - u_sample) / self.mu), time=time)
        else:
            li_dead_meme = self.read_feed(u_sample=(u_sample / (1 - self.mu)), time=time)
        if li_dead_meme:
            network.active_memes = [x for x in network.active_memes if x not in li_dead_meme]
    
    def connected(self, person):
        if person == self: return True
        else: return (person in self.friends) and (self in person.friends)
            
    def connect(self, person):
        if not self.connected(person):
            self.friends.append(person)
            person.connect(self)
            

class Network:
    def __init__(self, people=0, connexions=0, alpha=0, mu=0):
        self.alpha = alpha
        self.mu = mu
        self.people = []
        self.active_memes = []
        self.active_memes_count = []
        self.memes = []
        self.size = 0
        self.time = 0
        if connexions > people * (people - 1) / 2:
            raise Exception('Not enough people in the network to create '+str(connexions)+' different connexions')
        
        for i in range(people): self.add_person()
        self.random_connexions(connexions)

    def add_person(self):
        self.people.append(Person(feed=[], friends=[], alpha=self.alpha, mu=self.mu))
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
                
    def simulate(self, n_steps):
        t = time.time()
        for i in range(n_steps): self.next_timestep()
        print("runtime : {0:.2f} s".format(time.time()-t))

    def next_timestep(self):
        temp = random.randint(self.size)
        person = self.people[temp]
        person.action(self)
        self.time += 1
        self.active_memes_count.append(len(self.active_memes))

    def plot_shares(self):
        x = [meme.quality for meme in self.memes]
        y = [meme.shares for meme in self.memes]
        plt.scatter(x, y)
        plt.xlabel("Quality")
        plt.ylabel("Shares")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()

    def plot_views(self):
        x = [meme.quality for meme in self.memes]
        y = [meme.views for meme in self.memes]
        plt.scatter(x, y)
        plt.xlabel("Quality")
        plt.ylabel("Views")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()

    def plot_lifetimes(self):
        x = [meme.quality for meme in self.memes]
        y = [meme.end - meme.start for meme in self.memes]
        plt.scatter(x, y)
        plt.xlabel("Quality")
        plt.ylabel("Lifetime")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.show()

    def kendall_tau(self):
        x = [meme.quality for meme in self.memes]
        y = [meme.shares for meme in self.memes]
        (tau, p) = stats.kendalltau(x, y)
        return tau


if __name__ == '__main__':
    t = time.time()
    Alpha = 20
    Mu = 0.1
    n_steps = 10000
    n_people = 1000
    n_connexions = 100000
    
    net = Network(people=n_people, connexions=n_connexions, alpha=Alpha, mu=Mu)
    net.simulate(n_steps)
    net.plot_shares()
    net.plot_views()
    net.plot_lifetimes()
    
    memes_df = pd.DataFrame({'quality': [meme.quality for meme in net.memes],
                             'views': [meme.views for meme in net.memes],
                             'shares': [meme.shares for meme in net.memes],
                             'start': [meme.start for meme in net.memes],
                             'end': [meme.end for meme in net.memes]})

    memes_df['lifetime'] = memes_df['end'] - memes_df['start']
    print(memes_df)
    
    print("Kendall Tau : " + str(net.kendall_tau()))
            
    print("Exec time : "+str(time.time() - t)+" seconds")   
