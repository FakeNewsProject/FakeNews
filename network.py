# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 23:47:03 2017

@author: Rui Silva

"""

import numpy as np
import pandas as pd
from numpy import random
from random import shuffle


def quality_cdf(u):
    return u

def test(u):
    return "hello world"

    
class Meme:
    def __init__(self, quality, views, shares, start):
        self.quality = quality
        self.views = 0
        self.shares = 0
        self.start = start
        self.end = np.nan


class Person:
    def __init__(self, feed = [], friends = []):
        self.feed = []
        self.friends =[]
        
    def publish(self, network, quality, start, views=0, shares=0):
        mem = Meme(quality, views, shares, start)
        self.share(mem)
        network.memes.append(mem)
        
    def view(self, meme):
        meme.views += 1
        if meme in self.feed: self.feed.remove(meme)
        self.feed.append(meme)
        if len(self.feed) > alpha:
            self.feed.pop(0)
        
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
        
    def action(self, network, timestep):
        u_sample = random.uniform(0, 1)
        if u_sample >= 1-mu:
            self.publish(network=network, quality=quality_cdf((1 - u_sample) / mu), start=timestep)
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
                
    def simulate(self, n_steps):
        temp = random.randint(self.size, size=n_steps)
        for i in range(n_steps):
            if i % 200 == 0 : print(i),
            person = self.people[temp[i]]
            person.action(self, i)
            self.record_dead_memes(i)

    def record_dead_memes(self, timestep):
        for meme in self.memes:
            li_feed = [people.feed for people in self.people]
            li_in_feed = [meme in feed for feed in li_feed]
            is_alive = any(li_in_feed)
            if not is_alive and np.isnan(meme.end):
                meme.end = timestep


alpha = 10
mu = 0.05
people = 100
connexions = 2500
steps = 100

# if __name__ == "__main__":
#
#     network = Network(people, connexions)
#     network.simulate(steps)
#     print("memes count " + str(len(network.memes)))
#
#     data = dict(quality=[x.quality for x in network.memes], shares=[x.shares for x in network.memes],
#                 views=[x.views for x in network.memes], start=[x.start for x in network.memes],
#                 end=[x.end for x in network.memes])
#
#     df_memes = pd.DataFrame.from_dict(data)
#     df_memes.index = df_memes['quality']
#     df_memes = df_memes[['shares', 'views', 'start', 'end']]
#     print(df_memes.head())
#     print(df_memes.describe())

