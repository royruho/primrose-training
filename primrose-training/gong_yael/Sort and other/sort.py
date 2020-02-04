import numpy as np
import timeit

def swap(array,idx1,idx2):
    array[idx1],array[idx2] = array[idx2],array[idx1]
    return array

def partition(array, begin, end):
    low_index = begin
    high_index = end
    if high_index-low_index <=2 :
        return array
    while high_index > len(array)-1:
        high_index -= 1
    pivot_idx = int((high_index+low_index)/2)
    pivot = array[pivot_idx]
    swap(array,begin,pivot_idx)
    low_index+=1
    while low_index <= high_index and high_index >= 0 and high_index<= len(array)-1:
        if high_index >= low_index and array[high_index] >= pivot:
            high_index -= 1
        elif low_index <= high_index and array[low_index] < pivot:
            low_index += 1
        else :
            swap(array,low_index,high_index)
            low_index+=1
            high_index-=1
    swap(array,begin,high_index)
    #print('low idx', low_index, 'high idx', high_index, 'pivot', pivot)
    array =  partition(array,begin,low_index)
    array = partition(array,low_index,end)
    return array

def quick_sort(array):
    begin = 0
    end = len(array)-1
    sorted = partition(array,begin,end)
    return sorted

def bubble_sort(array):
    top_idx = len(array)
    repeat=True
    while repeat:
        repeat=False
        for i in range(1,top_idx):
            if array[i - 1] > array[i]:
                swap(array, i - 1, i)
                repeat=True
        top_idx -= 1
    return array

class node():
    def __init__(self,level =0, node_type = 'root'):
        self.node_type = node_type
        self.level = level
        self.val = None
        self.left = None
        self.right = None

    def insert(self,number):
        if self.val :
            if number > self.val :
                if not self.right:
                    self.right = node(self.level+1, node_type = 'right')
                    self.right.insert(number)
                else:
                    self.right.insert(number)
            else:
                if not self.left:
                    self.left = node(self.level+1, node_type = 'left')
                    self.left.insert(number)
                else:
                    self.left.insert(number)
        else :
            self.val = number
    def print_node(self):
        if self.left:
            self.left.print_node()
        if self.right:
            self.right.print_node()
        print ('level:',self.level, 'afilliation:', self.node_type, 'value:', self.val)

def set_like(list):
    set_ret = []
    for val in list:
        is_dup = False
        for set_val in set_ret:
            if val == set_val:
                is_dup = True
                break
        if is_dup == False:
            set_ret.append(val)
    return set_ret

def sorted_set(in_list):
    sorted_list = quick_sort(in_list)
    set_ret = [sorted_list[0]]
    for val in in_list[1:]:
        if val != set_ret[-1]:
            set_ret.append(val)
    return set_ret



if __name__ == '__main__':
    array=list(np.random.randint(0, 30000, 10000))
    start_time = timeit.default_timer()
    sorted1 = quick_sort(array.copy())
    print ('quick run for ', timeit.default_timer()-start_time)
    # start_time = timeit.default_timer()
    # sorted2= bubble_sort(array.copy())
    # print ('bubble run for ', timeit.default_timer()-start_time)
    start_time = timeit.default_timer()
    set1 = set_like(array.copy())
    print ('regular set run for ', timeit.default_timer()-start_time)
    print (len(set1))
    start_time = timeit.default_timer()
    set2 = sorted_set(array.copy())
    print ('sorted set set run for ', timeit.default_timer()-start_time)
    print (len(set2))

    a = node()
    a.insert(2)
    a.insert(3)
    a.insert(1)
    a.insert(4)
    a.insert(5)
    a.insert(0)
    a.insert(17)
