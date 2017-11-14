# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 23:47:03 2017

@author: Rui Silva

"""
from numpy import random
from random import shuffle
from random import choice
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import pandas as pd
import numpy as np
from numba import jit, prange, float64, int32

@jit(float64(float64))
def quality_cdf(u):
    return u

@jit(float64(float64, int32))
def share_proba(alpha, repetitions):
    #return ((repetitions + 5) ** 2) * (alpha ** (repetitions + 5))
    return np.exp(-((repetitions-40)/50)**2 )

@jit(float64(float64))
def time_distrib(l):                                                                                                                                     
    return 1 + random.poisson(l,1)

def publishTime():
    return 3000

def shareTime():
    return 1000


class Meme:
    def __init__(self, quality, views=0, shares=0, start=0):
        self.quality = quality
        self.views = views
        self.shares = shares
        self.ocurrences = 0
        self.start = start
        self.occurenceTime=dict()


class Person:
    def __init__(self,Id_pers ,mu=0.5, l = 0.1, alpha = 10):
        self.id_pers=Id_pers
        self.mu = mu
        self.feed = []
        self.friends = {}
        self.l = l
        self.alpha = alpha

    def publish(self, network, quality, time,real_time, shares=0, views=0):
        mem = Meme(quality, views, shares, start=time)
        self.share(mem,real_time)
        network.memes.append(mem)
        network.active_memes.append(mem)

    def view(self, meme,real_time, person):
        meme.views += 1
        n = len(self.feed)
        repetitions = 0
        for i in range(n):
            (mem, repet) = self.feed[i]
            if mem == meme:
                self.feed.remove((mem, repet))
                repetitions = repet
                meme.ocurrences -= 1
                break
        self.feed.append((meme, repetitions + 1))
        meme.ocurrences += 1
        meme.occurenceTime[self.id_pers]=real_time
        

    def share(self, meme,real_time):
        meme.shares += 1
        friends = list(self.friends.keys())
        n = len(friends)
        for i in prange(n):
            friends[i].view( meme,real_time,self)
        

    
    def read_feed(self, u_sample,real_time):
        quality_sum = 0
        temp = 0
        N = len(self.feed)
        n = min(time_distrib(self.l)[0], N)
        for i in list(range(n)):
            (meme, repetitions) = self.feed[N-i-1]
            quality_sum += meme.quality * share_proba(self.alpha, repetitions) #*connectivity
        for i in list(range(n)):
            (meme, repetitions) = self.feed[N - i - 1]
            temp += (meme.quality * share_proba(self.alpha, repetitions))/ quality_sum  #* connectivity)
            if u_sample <= temp:
                self.share(meme,real_time)

    def action(self, network):
        u_sample = random.uniform(0, 1)
        time = network.time
        real_time=network.real_time
        if u_sample >= 1 - self.mu:
            lapstime=publishTime()
            network.real_time+=lapstime
            self.publish(network=network, quality=quality_cdf((1 - u_sample) / self.mu), time=time,real_time=real_time+lapstime)
        else:
            lapstime=shareTime()
            network.real_time+=lapstime
            self.read_feed(u_sample=(u_sample / (1 - self.mu)),real_time=real_time+lapstime)

    def connected(self, person):
        if person == self:
            return True
        else:
            return person in self.friends

    def connect(self, person):
        if not self.connected(person):
            self.friends[person] = 0
            person.friends[self] = 0


class Network:   
    def __init__(self, people=0, n_connexions=0, alpha=0, mu=0.1, l = 0.1,num_people=0):
        self.num_people=num_people
        self.alpha = alpha
        self.mu = mu
        self.people = []
        self.active_memes = []
        self.active_memes_count = []
        self.entropies = []
        self.memes = []
        self.size = 0
        self.time = 0
        self.real_time=0
        self.l = l
        if n_connexions > people * (people - 1) / 2:
            raise Exception('Not enough people in the network to create ' + str(n_connexions) + ' different connexions')

        for i in range(people): self.add_person()
        self.random_connexions(n_connexions)

    def add_person(self):
        self.people.append(Person(self.num_people ,mu=self.mu, l = self.l, alpha = self.alpha))
        self.num_people+=1
        self.size += 1

    def connected(self, person1, person2):
        person1.connected(person2)

    def connect(self, person1, person2):
        person1.connect(person2)

    def pagerank(self, person1, person2, tau, n):
        count = 0.
        for i in range(n):
            if person2 == self.random_walk(person1, tau): count += 1.
        return count / n

    def random_walk(self, person, tau):
        u = random.uniform(0, 1)
        if u < tau:
            return person
        else:
            person2 = choice(list(person.friends.keys()))
            return self.random_walk(person2, tau)
        

    def calc_connectivities(self, tau, n):
        for person1 in self.people:
            for person2 in person1.friends:
                person1.friends[person2] = self.pagerank(person1, person2, tau, n)

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
        for i in range(n_steps): self.next_timestep()
        print("Total time is ",np.floor(self.real_time/(3600*24)),"days")

    def next_timestep(self):
        temp = random.randint(self.size)
        person = self.people[temp]
        person.action(self)
        self.time += 1
        self.active_memes_count.append(len(self.active_memes))
        self.entropies.append(self.entropy())

    def plot_entropy(self):
        x = range(self.time)
        y = self.entropies
        plt.scatter(x, y)
        plt.xlabel("Time")
        plt.ylabel("Entropy")
        plt.xlim(xmin=0, xmax=self.time)
        plt.ylim(ymin=0)
        plt.show()

    def plot_activememecount(self):
        x = range(self.time)
        y = self.active_memes_count
        plt.scatter(x, y)
        plt.xlabel("Time")
        plt.ylabel("Active Memes")
        plt.xlim(xmin=0, xmax=self.time)
        plt.ylim(ymin=0)
        plt.show()

    def plot_shares(self):
        x = [meme.quality for meme in self.memes]
        y = [meme.shares for meme in self.memes]
        plt.scatter(x, y)
        plt.xlabel("Quality")
        plt.ylabel("Shares")
        plt.xlim(xmin=0, xmax=1)
        plt.ylim(ymin=0)
        plt.show()

    def plot_views(self):
        x = [meme.quality for meme in self.memes]
        y = [meme.views for meme in self.memes]
        plt.scatter(x, y)
        plt.xlabel("Quality")
        plt.ylabel("Views")
        plt.xlim(xmin=0, xmax=1)
        plt.ylim(ymin=0)
        plt.show()

    def kendall_tau(self):
        x = [meme.quality for meme in self.memes]
        y = [meme.shares for meme in self.memes]
        (tau, p) = stats.kendalltau(x, y)
        return tau

    def plot_shares_byquantiles(self, nb_quantiles):
        df = pd.DataFrame({'Quality': [meme.quality for meme in self.memes],
                           'Shares': [meme.shares for meme in self.memes]}).dropna()
        df['Quality'] = ((1 - df['Quality'].rank(method='min', na_option='keep', ascending=False,
                                                 pct=True)) * nb_quantiles).apply(int) + 1
        plot = df.groupby('Quality').mean()
        plot.plot(kind='bar')
        plt.show()

    def plot_views_byquantiles(self, nb_quantiles):
        df = pd.DataFrame({'Quality': [meme.quality for meme in self.memes],
                           'Views': [meme.views for meme in self.memes], }).dropna()
        df['Quality'] = ((1 - df['Quality'].rank(method='min', na_option='keep', ascending=False,
                                                 pct=True)) * nb_quantiles).apply(int) + 1
        plot = df.groupby('Quality').mean()
        plot.plot(kind='bar')
        plt.show()

    def entropy(self):
        sum = np.sum([meme.ocurrences for meme in self.active_memes])
        attention_list = [meme.ocurrences / sum for meme in self.active_memes]
        entropy = -np.sum(attention * np.log(attention) for attention in attention_list)
        return entropy


if __name__ == '__main__':
    t = time.time()
    Mu = 0.2
    l = 20
    alpha = 0.95
    n_steps = 10000
    n_people = 1000
    tau = 0.5

    n_connexions = 10000

    net = Network(people=n_people, n_connexions=n_connexions, mu=Mu, l=l, alpha=alpha)
    #net.calc_connectivities(tau, 10000)
    net.simulate(n_steps)

    net.plot_shares_byquantiles(40)
    net.plot_shares()
    net.plot_entropy()
    net.plot_views_byquantiles(40)
    net.plot_views()

    memes_df = pd.DataFrame({'quality': [meme.quality for meme in net.memes],
                             'views': [meme.views for meme in net.memes],
                             'shares': [meme.shares for meme in net.memes],
                             'start': [meme.start for meme in net.memes]})


    print("Kendall Tau : " + str(net.kendall_tau()))

    print("Exec time : " + str(time.time() - t) + " seconds")
