'''
Created on Dec 30, 2024

@author: pglauner
'''

import math
import mmh3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, number_expected_elements, fp_prob):
        self._number_expected_elements = number_expected_elements
        self._fp_prob = fp_prob
        self._size = self._get_size()
        self._hash_count = self._get_hash_count()
        self._bitarray = bitarray(self._size)
        self._bitarray.setall(0)
    def add(self, item):
        for i in range(self._hash_count):
            # With different seed, indexes are spread out
            index = mmh3.hash(item, i) % self._size
            self._bitarray[index] = True
    def check(self, item):
        for i in range(self._hash_count):
            index = mmh3.hash(item, i) % self._size
            if not self._bitarray[index]:
                return False
        return True
    def _get_size(self):
        n = -(self._number_expected_elements*math.log(self._fp_prob))/(math.log(2)**2)
        return math.ceil(n)
    def _get_hash_count(self):
        k = (self._size/self._number_expected_elements)*math.log(2)
        return math.ceil(k)

if __name__ == '__main__':
    from random import shuffle
    m = 20 # number of items to add
    p = 0.05 # false positive probability

    bloom_filter = BloomFilter(m, p)
    print('Size of bit array: {}'.format(bloom_filter._size))
    print('False positive Probability: {}'.format(bloom_filter._fp_prob))
    print('Number of hash functions: {}'.format(bloom_filter._hash_count))

    words_present = ['abound','abounds','abundance','abundant','accessible'
                'bloom','blossom','bolster','bonny','bonus','bonuses',
                'coherent','cohesive','colorful','comely','comfort',
                'gems','generosity','generous','generously','genial']
    words_absent = ['bluff','cheater','hate','war','humanity',
               'racism','hurt','nuke','gloomy','facebook',
               'geeksforgeeks','twitter']

    for item in words_present:
        bloom_filter.add(item)

    shuffle(words_present)
    shuffle(words_absent)

    test_words = words_present[:10] + words_absent
    shuffle(test_words)
    for word in test_words:
        if bloom_filter.check(word):
            if word in words_absent:
                print("'{}' is a false positive!".format(word))
            else:
                print("'{}' is probably present!".format(word))
        else:
            print("'{}' is definitely not present!".format(word))
