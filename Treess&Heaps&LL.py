# Functional implementation of a tree Pythonds

# https://realpython.com/python-magic-methods/ --Magic methods very imprtant
# https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python --print tree level by level (Tree visualization)

# https://www.pythontutorial.net/python-built-in-functions/python-next/
### linux for command line ###
# https://towardsdatascience.com/start-using-linux-commands-to-quick-analyze-structured-data-not-pandas-f63065842269
# https://towardsdatascience.com/start-using-linux-commands-to-quick-analyze-structured-data-not-pandas-f63065842269

############### References ###################
# accessing the variables of other classes
# https://dnmtechs.com/accessing-variables-between-classes-in-python-3/

############## Linked List Implementation start##############
# initialize an empty class

# https://www.geeksforgeeks.org/difference-between-binary-tree-and-binary-search-tree/

import unittest
# from .bst import BinarySearchTree,TreeNode
# list representation
myTree = ['a',  # root
          ['b',  # left subtree
           ['d', [], []],
           ['e', [], []]],
          ['c',  # right subtree
           ['f', [], []],
           []]
          ]
# sample 2
myt1=['r',['lr0',['lr1',[],[]],[]],['rr0',[],['rr1',[],[]]]]
# ['lr0', ['lr1', [], []], []] # myt1[1]
# ['rr0', [], ['rr1', [], []]] # myt1[2]

########### Tree functional  ###############################
def BinaryTree(r):
    return [r, [], []]


def insertLeft(root, newBranch):
    t = root.pop(1)
    if len(t) > 1:
        root.insert(1, [newBranch, t, []])
    else:
        root.insert(1, [newBranch, [], []])
    return root


def insertRight(root, newBranch):
    t = root.pop(2)
    if len(t) > 1:
        root.insert(2, [newBranch, [], t])
    else:
        root.insert(2, [newBranch, [], []])
    return root


def getRootVal(root):
    return root[0]


def setRootVal(root, newVal):
    root[0] = newVal


def getLeftChild(root):
    return root[1]


def getRightChild(root):
    return root[2]


def inorder(tree):
    if tree != None:
        inorder(tree.getLeftChild())
        print(tree.getRootVal())
        inorder(tree.getRightChild())



r = BinaryTree(3)
insertLeft(r, 4)
insertLeft(r, 5)
insertRight(r, 6)
insertRight(r, 7)
l = getLeftChild(r)
print(l)

setRootVal(l, 9)
print(r)
insertLeft(l, 11)
print(r)
print(getRightChild(getRightChild(r)))


########### Tree functional end ###############################

class TreeNode:
    def __init__(self, key, val, left=None, right=None, parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent
        self.balanceFactor = 0

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self, key, value, lc, rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self

    def findSuccessor(self):
        succ = None
        if self.hasRightChild():
            succ = self.rightChild.findMin()
        else:
            if self.parent:
                if self.isLeftChild():
                    succ = self.parent
                else:
                    self.parent.rightChild = None
                    succ = self.parent.findSuccessor()
                    self.parent.rightChild = self
        return succ

    def spliceOut(self):
        if self.isLeaf():
            if self.isLeftChild():
                self.parent.leftChild = None
            else:
                self.parent.rightChild = None
        elif self.hasAnyChildren():
            if self.hasLeftChild():
                if self.isLeftChild():
                    self.parent.leftChild = self.leftChild
                else:
                    self.parent.rightChild = self.leftChild
                self.leftChild.parent = self.parent
            else:
                if self.isLeftChild():
                    self.parent.leftChild = self.rightChild
                else:
                    self.parent.rightChild = self.rightChild
                self.rightChild.parent = self.parent

    def findMin(self):
        current = self
        while current.hasLeftChild():
            current = current.leftChild
        return current

    def __iter__(self):
        """The standard inorder traversal of a binary tree."""
        if self:
            if self.hasLeftChild():
                for elem in self.leftChild:
                    yield elem
            yield self.key
            if self.hasRightChild():
                for elem in self.rightChild:
                    yield elem


class BinaryTree: # insert logic same as the functional implementation
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t


    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key

# Binary search tree implementation Pythonds

class BinarySearchTree:
    '''
    Author:  Brad Miller
    Date:  1/15/2005
    Description:  Imlement a binary search tree with the following interface
                  functions:
                  __contains__(y) <==> y in x
                  __getitem__(y) <==> x[y]
                  __init__()
                  __len__() <==> len(x)
                  __setitem__(k,v) <==> x[k] = v
                  clear()
                  get(k)
                  items()
                  keys()
                  values()
                  put(k,v)
                  in
                  del <==>
    '''

    def __init__(self):
        self.root = None
        self.size = 0

    def put(self, key, val):
        if self.root:
            self._put(key, val, self.root)
        else:
            self.root = TreeNode(key, val)
        self.size = self.size + 1

    def _put(self, key, val, currentNode):
        if key < currentNode.key: # this recursion can be achieved by the swapping logic too
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)
            else:
                currentNode.leftChild = TreeNode(key, val, parent=currentNode)
        else:
            if currentNode.hasRightChild():
                self._put(key, val, currentNode.rightChild)
            else:
                currentNode.rightChild = TreeNode(key, val, parent=currentNode)

    def __setitem__(self, k, v):
        self.put(k, v)

    def get(self, key):
        if self.root:
            res = self._get(key, self.root)
            if res:
                return res.payload
            else:
                return None
        else:
            return None

    def _get(self, key, currentNode):
        if not currentNode:
            return None
        elif currentNode.key == key:
            return currentNode
        elif key < currentNode.key:
            return self._get(key, currentNode.leftChild)
        else:
            return self._get(key, currentNode.rightChild)

    def __getitem__(self, key):
        res = self.get(key)
        if res:
            return res
        else:
            raise KeyError('Error, key not in tree')

    def __contains__(self, key):
        if self._get(key, self.root):
            return True
        else:
            return False

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.root.__iter__()

    def delete(self, key):
        if self.size > 1:
            nodeToRemove = self._get(key, self.root)
            if nodeToRemove:
                self.remove(nodeToRemove)
                self.size = self.size - 1
            else:
                raise KeyError('Error, key not in tree')
        elif self.size == 1 and self.root.key == key:
            self.root = None
            self.size = self.size - 1
        else:
            raise KeyError('Error, key not in tree')

    def __delitem__(self, key):
        self.delete(key)

    def remove(self, currentNode):
        if currentNode.isLeaf():  # leaf
            if currentNode == currentNode.parent.leftChild:
                currentNode.parent.leftChild = None
            else:
                currentNode.parent.rightChild = None
        elif currentNode.hasBothChildren():  # interior
            succ = currentNode.findSuccessor()
            succ.spliceOut()
            currentNode.key = succ.key
            currentNode.payload = succ.payload
        else:  # this node has one child
            if currentNode.hasLeftChild():
                if currentNode.isLeftChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.leftChild
                elif currentNode.isRightChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.leftChild
                else:
                    currentNode.replaceNodeData(currentNode.leftChild.key,
                                                currentNode.leftChild.payload,
                                                currentNode.leftChild.leftChild,
                                                currentNode.leftChild.rightChild)
            else:
                if currentNode.isLeftChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.rightChild
                elif currentNode.isRightChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.rightChild
                else:
                    currentNode.replaceNodeData(currentNode.rightChild.key,
                                                currentNode.rightChild.payload,
                                                currentNode.rightChild.leftChild,
                                                currentNode.rightChild.rightChild)

    def inorder(self):
        self._inorder(self.root)

    def _inorder(self, tree):
        if tree != None:
            self._inorder(tree.leftChild)
            print(tree.key)
            self._inorder(tree.rightChild)

    def postorder(self):
        self._postorder(self.root)

    def _postorder(self, tree):
        if tree:
            self._postorder(tree.leftChild)
            self._postorder(tree.rightChild)
            print(tree.key)

    def preorder(self):
        self._preorder(self, self.root)

    def _preorder(self, tree):
        if tree:
            print(tree.key)
            self._preorder(tree.leftChild)
            self._preorder(tree.rightChild)
# https://github.com/bnmnetp/pythonds/blob/master/trees/balance.py

########## AVL Tree ###########

# !/bin/env python3.1
# Bradley N. Miller, David L. Ranum
# Introduction to Data Structures and Algorithms in Python
# Copyright 2005, 2010

class AVLTree(BinarySearchTree):
    '''
    Author:  Brad Miller
    Date:  1/15/2005
    Description:  Imlement a binary search tree with the following interface
          functions:
                  __contains__(y) <==> y in x
                  __getitem__(y) <==> x[y]
                  __init__()
                  __len__() <==> len(x)
                  __setitem__(k,v) <==> x[k] = v
                  clear()
                  get(k)
                  has_key(k)
                  items()
                  keys()
                  values()
                  put(k,v)
    '''
    # understand the put overload
    def _put(self, key, val, currentNode): # _put is a helper function of put() which is overloaded using []
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)
            else:
                currentNode.leftChild = TreeNode(key, val, parent=currentNode)
                self.updateBalance(currentNode.leftChild)
        else:
            if currentNode.hasRightChild():
                self._put(key, val, currentNode.rightChild)
            else:
                currentNode.rightChild = TreeNode(key, val, parent=currentNode)
                self.updateBalance(currentNode.rightChild)

    def updateBalance(self, node):
        if node.balanceFactor > 1 or node.balanceFactor < -1:
            self.rebalance(node)
            return
        if node.parent != None:
            if node.isLeftChild():
                node.parent.balanceFactor += 1
            elif node.isRightChild():
                node.parent.balanceFactor -= 1

            if node.parent.balanceFactor != 0:
                self.updateBalance(node.parent)

    def rebalance(self, node):
        if node.balanceFactor < 0:
            if node.rightChild.balanceFactor > 0:
                # Do an LR Rotation
                self.rotateRight(node.rightChild)
                self.rotateLeft(node)
            else:
                # single left
                self.rotateLeft(node)
        elif node.balanceFactor > 0:
            if node.leftChild.balanceFactor < 0:
                # Do an RL Rotation
                self.rotateLeft(node.leftChild)
                self.rotateRight(node)
            else:
                # single right
                self.rotateRight(node)

    def rotateLeft(self, rotRoot):
        newRoot = rotRoot.rightChild
        rotRoot.rightChild = newRoot.leftChild
        if newRoot.leftChild != None:
            newRoot.leftChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isLeftChild():
                rotRoot.parent.leftChild = newRoot
            else:
                rotRoot.parent.rightChild = newRoot
        newRoot.leftChild = rotRoot
        rotRoot.parent = newRoot
        rotRoot.balanceFactor = rotRoot.balanceFactor + 1 - min(newRoot.balanceFactor, 0)
        newRoot.balanceFactor = newRoot.balanceFactor + 1 + max(rotRoot.balanceFactor, 0)

    def rotateRight(self, rotRoot):
        newRoot = rotRoot.leftChild
        rotRoot.leftChild = newRoot.rightChild
        if newRoot.rightChild != None:
            newRoot.rightChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isRightChild():
                rotRoot.parent.rightChild = newRoot
            else:
                rotRoot.parent.leftChild = newRoot
        newRoot.rightChild = rotRoot
        rotRoot.parent = newRoot
        rotRoot.balanceFactor = rotRoot.balanceFactor - 1 - max(newRoot.balanceFactor, 0)
        newRoot.balanceFactor = newRoot.balanceFactor - 1 + min(rotRoot.balanceFactor, 0)

print(myTree)
print('left subtree = ', myTree[1])


r = BinaryTree(3)
insertLeft(r,4)
insertLeft(r,5)
insertRight(r,6)
insertRight(r,7)
l = getLeftChild(r)
print(l)

setRootVal(l,9)
print(r)
insertLeft(l,11)
print(r)
print(getRightChild(getRightChild(r)))
# extend it to functional traversals --very imp

# (possibility of a non recurcive implementation of tree traversals)

t = BinaryTree(7)
t.insertLeft(3)
t.insertRight(9)
inorder(t)
x = BinaryTree('*')
x.insertLeft('+')
l = x.getLeftChild()
l.insertLeft(4)
l.insertRight(5)
x.insertRight(7)
# print(printexp(x))
# print(postordereval(x))
# print(height(x))
t.insertRight(19)

t.insertLeft(31)
t.insertRight(8)

mytree = BinarySearchTree()
mytree[3]="red"
mytree[4]="blue"
mytree[6]="yellow"
mytree[2]="at"

########### traversals ##########
mytree.preorder()# traversal preorder
mytree.postorder()# traversal postorder
######## traversals end ##########

print(mytree)
print(mytree[6])
print(mytree[2])

########## AVL Tree ###########
av=AVLTree()
av.put('a',4)
av.put('a',17)
av.put('a',41)
av.put('a',34)
av.put('a',12)
av.put('a',44)
av.put('a',21)
av.put('a',14)


bst = AVLTree()
bst.put(30, 'a')
bst.put(50, 'b')
bst.put(40, 'c')
assert bst.root.key == 40

bst.put(50, 'a')
bst.put(30, 'b')
bst.put(40, 'c')
assert bst.root.key == 40

bst.put(50, 'a')
bst.put(30, 'b')
bst.put(70, 'c')
bst.put(80, 'c')
bst.put(60, 'd')
bst.put(90, 'e')
assert bst.root.key == 70
bst.put(40, 'a')
bst.put(30, 'b')
bst.put(50, 'c')
bst.put(45, 'd')
bst.put(60, 'e')
bst.put(43, 'f')

bst.put(5, 'g')
bst.put(4, 'h')
bst.put(6, 'i')
bst.put(2, 'j')
########## AVL Tree ###########


################## Binary Tree Test Cases ##################

class BinaryTreeTests(unittest.TestCase):
    def setUp(self):
        self.bst = AVLTree()

    def testAuto1(self):
        self.bst.put(30, 'a')
        self.bst.put(50, 'b')
        self.bst.put(40, 'c')
        assert self.bst.root.key == 40

    def testAuto2(self):
        self.bst.put(50, 'a')
        self.bst.put(30, 'b')
        self.bst.put(40, 'c')
        assert self.bst.root.key == 40

    def testAuto3(self):
        self.bst.put(50, 'a')
        self.bst.put(30, 'b')
        self.bst.put(70, 'c')
        self.bst.put(80, 'c')
        self.bst.put(60, 'd')
        self.bst.put(90, 'e')
        assert self.bst.root.key == 70

    def testAuto3(self):
        self.bst.put(40, 'a')
        self.bst.put(30, 'b')
        self.bst.put(50, 'c')
        self.bst.put(45, 'd')
        self.bst.put(60, 'e')
        self.bst.put(43, 'f')

        assert self.bst.root.key == 45
        assert self.bst.root.leftChild.key == 40
        assert self.bst.root.rightChild.key == 50
        assert self.bst.root.balanceFactor == 0
        assert self.bst.root.leftChild.balanceFactor == 0
        assert self.bst.root.rightChild.balanceFactor == -1

    def testAuto4(self):
        self.bst.put(40, 'a')
        self.bst.put(30, 'b')
        self.bst.put(50, 'c')
        self.bst.put(10, 'd')
        self.bst.put(35, 'e')
        self.bst.put(37, 'f')

        assert self.bst.root.key == 35
        assert self.bst.root.leftChild.key == 30
        assert self.bst.root.rightChild.key == 40
        assert self.bst.root.balanceFactor == 0
        assert self.bst.root.leftChild.balanceFactor == 1
        assert self.bst.root.rightChild.balanceFactor == 0


# if __name__ == '__main__':
import platform

print(platform.python_version())
unittest.main()

#### try this later but immediately ########
assert bst.root.key == 45
assert bst.root.leftChild.key == 40
assert bst.root.rightChild.key == 50
assert bst.root.balanceFactor == 0
assert bst.root.leftChild.balanceFactor == 0
assert bst.root.rightChild.balanceFactor == -1
#### try this later but immediately end ########

# Local Variables:
# py-which-shell: "python3"
# End:

def printTree(node, level=0):
    if node != None:
        printTree(node.leftChild, level + 1)
        print(' ' * 4 * level + '-> ' + str(node.key) + str(node.payload))
        printTree(node.rightChild, level + 1)

# printTree(mytree.root,level=0)
printTree(bst.root) # check with .root
print('Tree printed')

################## Binary Tree Test Cases end###############

# work on putting traversals

# queue and stack classes and a little browsing on search

# Priority Queue Implementation

# https://github.com/bnmnetp/pythonds/blob/master/graphs/priorityQueue.py

#  inserting into a list is O(n) and sorting a list is O(nlogn) so the total time for n insertions is O(nlogn) which is not good.We can do better. The classic way to implement a priority queue is using a data structure called a binary heap. A binary heap will allow us both enqueue and dequeue items in O(logn).
############# linked list complete end ################
# https://github.com/bnmnetp/pythonds/blob/master/graphs/priorityQueue.py
'''
A priority queue is an abstract datatype. It is a shorthand way of describing a particular interface and behavior, and says nothing about the underlying implementation.

A heap is a data structure. It is a name for a particular way of storing data that makes certain operations very efficient.

It just so happens that a heap is a very good data structure to implement a priority queue, because the operations which are made efficient by the heap data strucure are the operations that the priority queue interface needs.

# Property:Each Node greater than or equal to its children
# eg: 0 -> 11 -> 4 -> 5 -> 2 -> 3

# Heap perculate down and up implementation works on the property of LIST representation of tree
'''
class PriorityQueue:
    def __init__(self):
        # self.heapArray = [(0, 0)]
        self.heapArray = [0]
        self.currentSize = 0

    def buildHeap(self, alist):
        self.currentSize = len(alist)
        # self.heapArray = [(0, 0)]
        for i in alist:
            self.heapArray.append(i)
        i = len(alist) // 2
        while (i > 0):
            self.percDown(i)
            i = i - 1

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            # if self.heapArray[i][0] > self.heapArray[mc][0]:
            if self.heapArray[i] > self.heapArray[mc]:
                tmp = self.heapArray[i]
                self.heapArray[i] = self.heapArray[mc]
                self.heapArray[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 > self.currentSize:
            return -1
        else:
            if i * 2 + 1 > self.currentSize:
                return i * 2
            else:
                # if self.heapArray[i * 2][0] < self.heapArray[i * 2 + 1][0]:commented since its not a tuple
                if self.heapArray[i * 2] < self.heapArray[i * 2 + 1]:
                    return i * 2
                else:
                    return i * 2 + 1

    def percUp(self, i):
        while i // 2 > 0:
            # if self.heapArray[i][0] < self.heapArray[i // 2][0]:
            if self.heapArray[i] < self.heapArray[i // 2]:
                tmp = self.heapArray[i // 2]
                self.heapArray[i // 2] = self.heapArray[i]
                self.heapArray[i] = tmp
            i = i // 2

    def add(self, k):
        self.heapArray.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def delMin(self):
        # retval = self.heapArray[1][1] # this is for tuple
        retval = self.heapArray[1]
        self.heapArray[1] = self.heapArray[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapArray.pop()
        self.percDown(1)
        return retval

    def isEmpty(self):
        if self.currentSize == 0:
            return True
        else:
            return False

    def decreaseKey(self, val, amt): # is this the same as changePriority??
        # this is a little wierd, but we need to find the heap thing to decrease by
        # looking at its value
        done = False
        i = 1
        myKey = 0
        while not done and i <= self.currentSize:
            if self.heapArray[i][1] == val:
                done = True
                myKey = i
            else:
                i = i + 1
        if myKey > 0:
            self.heapArray[myKey] = (amt, self.heapArray[myKey][1])
            self.percUp(myKey)

    def __contains__(self, vtx):
        for pair in self.heapArray:
            if pair[1] == vtx:
                return True
        return False

    def __str__(self):# this is a additional method to list the heap as PQ list
        return str(self.heapArray)

    def insert(self, k):
        self.add(k)

    def findMin(self):
        # return self.heapArray[1][1] # this is for tuple
        return self.heapArray[1]

##### Priority Queue Implementation end ######

# BinHeap Implementation
# Pythonds
# https://runestone.academy/runestone/books/published/pythonds/Trees/BinaryHeapImplementation.html

#### this is same as priority q in this case
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percUp(self, i):
        while i // 2 > 0:

            if self.heapList[i] < self.heapList[i // 2]:

                tmp = self.heapList[i // 2]

                self.heapList[i // 2] = self.heapList[i]

                self.heapList[i] = tmp

            i = i // 2

    def percDown(self, i):

        while (i * 2) <= self.currentSize:

            mc = self.minChild(i)

            if self.heapList[i] > self.heapList[mc]:

                tmp = self.heapList[i]

                self.heapList[i] = self.heapList[mc]

                self.heapList[mc] = tmp

            i = mc

    def minChild(self, i):

        if i * 2 + 1 > self.currentSize:

            return i * 2

        else:

            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:

                return i * 2

            else:

                return i * 2 + 1

    def delMin(self):

        retval = self.heapList[1]

        self.heapList[1] = self.heapList[self.currentSize]

        self.currentSize = self.currentSize - 1

        self.heapList.pop()

        self.percDown(1)

        return retval

    def buildHeap(self, alist):

        i = len(alist) // 2

        self.currentSize = len(alist)

        self.heapList = [0] + alist[:]

        while (i > 0):

            self.percDown(i)

            i = i - 1
p=PriorityQueue()
p.buildHeap([(9,2),(5,3),(6,1),(2,9),(3,0)])
p.add((15,3))
print(p.heapArray)
p.decreaseKey(0,3)
print(p.heapArray)
p.decreaseKey(0,3)
print(p.heapArray)
p.decreaseKey(2,1)
print(p.heapArray)
p.delMin()
print(p)

bh = BinHeap() # replace BinHeap with PriorityQ
bh.insert(5)
bh.insert(7)
bh.insert(3)
bh.insert(11)

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())


############## Heap Implementation end ################

############# linked list implemenation  ################

class node:
    def __init__(self,nodevalue):
        self.data = nodevalue
        self.next = None

    def __str__(self):
        return str(self.data)

    def setData(self, nodevalue):# confusion on return value which is getValue
        self.data = nodevalue

    def getData(self):
        return self.data

    def setNext(self, nodevalue):
        self.next = nodevalue

    def getNext(self):
        return self.next

    def hasNext(self): # this is a additional method
        return self.next != None

    def insert(self, nodevalue): # this is an additional method
        pass


class LinkedList():
    def __init__(self, head=None):
        self.head = head

    def insert(self, data):
        new_node = node(data)
        new_node.set_next(self.head)
        self.head = new_node

    def size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.get_next()
        return count

    def search(self, data):
        current = self.head
        found = False
        while current and found is False:
            if current.get_data() == data:
                found = True
            else:
                current = current.get_next()
        if current is None:
            raise ValueError("Data not in list")
        return current
   ######### added later from a different example ########
    def delete(self, data):
        current = self.head
        previous = None
        found = False
        while current and found is False:
            if current.get_data() == data:
                found = True
            else:
                previous = current
                current = current.get_next()
        if current is None:
            raise ValueError("Data not in list")
        if previous is None:
            self.head = current.get_next()
        else:
            previous.set_next(current.get_next())

    def append(self, data):
        new_node = node(data)
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def prepend(self, data):
        new_node = node(data)
        new_node.next = self.head
        self.head = new_node

    def delete_with_value(self, data):
        if not self.head:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        current_node = self.head
        while current_node.next and current_node.next.data != data:
            current_node = current_node.next
        if current_node.next:
            current_node.next = current_node.next.next

    def print_list(self):
        current_node = self.head
        while current_node:
            print(current_node.data, end=" -> ")
            current_node = current_node.next
        print("None")

    ######### added later from a different example end########
# Example usage:
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.prepend(0)
ll.print_list()  # Output: 0 -> 1 -> 2 -> 3 -> None
ll.delete_with_value(2)
ll.print_list()  # Output: 0 -> 1 -> 3 -> None

# Create a LinkedList class

# Create a Node class to create a node
class Node:
    def __init__(self,nodevalue):
        self.data = nodevalue
        self.next = None

    def __str__(self):
        return str(self.data)

    def setData(self, nodevalue):# confusion on return value which is getValue
        self.data = nodevalue

    def getData(self):
        return self.data

    def setNext(self, nodevalue):
        self.next = nodevalue

    def getNext(self):
        return self.next

    def hasNext(self): # this is a additional method
        return self.next != None

    def insert(self, nodevalue): # this is an additional method
        pass

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None


class LinkedList1:
    def __init__(self):
        self.head = None

    # Method to add a node at the beginning of the LL
    def insertAtBegin(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    # Method to add a node at any index
    # Indexing starts from 0.
    def insertAtIndex(self, data, index):
        if index == 0:
            self.insertAtBegin(data)
            return

        position = 0
        current_node = self.head
        while current_node is not None and position + 1 != index:
            position += 1
            current_node = current_node.next

        if current_node is not None:
            new_node = Node(data)
            new_node.next = current_node.next
            current_node.next = new_node
        else:
            print("Index not present")

    # Method to add a node at the end of LL
    def insertAtEnd(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return

        current_node = self.head
        while current_node.next:
            current_node = current_node.next

        current_node.next = new_node

    # Update node at a given position
    def updateNode(self, val, index):
        current_node = self.head
        position = 0
        while current_node is not None and position != index:
            position += 1
            current_node = current_node.next

        if current_node is not None:
            current_node.data = val
        else:
            print("Index not present")

    # Method to remove first node of linked list
    def remove_first_node(self):
        if self.head is None:
            return

        self.head = self.head.next

    # Method to remove last node of linked list
    def remove_last_node(self):
        if self.head is None:
            return

        # If there's only one node
        if self.head.next is None:
            self.head = None
            return

        # Traverse to the second last node
        current_node = self.head
        while current_node.next and current_node.next.next:
            current_node = current_node.next

        current_node.next = None

    # Method to remove a node at a given index
    def remove_at_index(self, index):
        if self.head is None:
            return

        if index == 0:
            self.remove_first_node()
            return

        current_node = self.head
        position = 0
        while current_node is not None and current_node.next is not None and position + 1 != index:
            position += 1
            current_node = current_node.next

        if current_node is not None and current_node.next is not None:
            current_node.next = current_node.next.next
        else:
            print("Index not present")

    # Method to remove a node from the linked list by its data
    def remove_node(self, data):
        current_node = self.head

        # If the node to be removed is the head node
        if current_node is not None and current_node.data == data:
            self.remove_first_node()
            return

        # Traverse and find the node with the matching data
        while current_node is not None and current_node.next is not None:
            if current_node.next.data == data:
                current_node.next = current_node.next.next
                return
            current_node = current_node.next

        # If the data was not found
        print("Node with the given data not found")

    # Print the size of the linked list
    def sizeOfLL(self):
        size = 0
        current_node = self.head
        while current_node:
            size += 1
            current_node = current_node.next
        return size

    # Print the linked list
    def printLL(self):
        current_node = self.head
        while current_node:
            print(current_node.data)
            current_node = current_node.next


# create a new linked list
llist = LinkedList1()

# add nodes to the linked list
llist.insertAtEnd('a')
llist.insertAtEnd('b')
llist.insertAtBegin('c')
llist.insertAtEnd('d')
llist.insertAtIndex('g', 2)

# print the linked list
print("Node Data:")
llist.printLL()

# remove nodes from the linked list
print("\nRemove First Node:")
llist.remove_first_node()
llist.printLL()

print("\nRemove Last Node:")
llist.remove_last_node()
llist.printLL()

print("\nRemove Node at Index 1:")
llist.remove_at_index(1)
llist.printLL()

# print the linked list after all removals
print("\nLinked list after removing a node:")
llist.printLL()

print("\nUpdate node Value at Index 0:")
llist.updateNode('z', 0)
llist.printLL()

print("\nSize of linked list:", llist.sizeOfLL())


class UnorderedList:

    def __init__(self):
        self.head = None

    def isEmpty(self):
        return self.head == None

    def add(self, item):
        temp = node(item)
        temp.setNext(self.head)
        self.head = temp

    def size(self):
        current = self.head
        count = 0
        while current != None:
            count = count + 1
            current = current.getNext()
        return count

    def search(self,item):
        current = self.head
        found = False
        while current != None and not found:
            if current.getData() == item:
                found = True
            else:
                current = current.getNext()
        return found

    def remove(self,item):
        current = self.head
        previous = None
        found = False
        while not found:
            if current.getData() == item:
                found = True
            else:
                previous = current
                current = current.getNext()
        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())

    def insert(self, item, position): # this is an additional method
        current = self.head
        previous = None
        count = 0
        while current != None and count < position:
            previous = current
            current = current.getNext()
            count = count + 1
        temp = node(item)
        if previous == None:
            temp.setNext(self.head)
            self.head = temp
        else:
            temp.setNext(current)
            previous.setNext(temp)

    def append(self, item):
        current = self.head
        previous = None
        while current != None:
            previous = current
            current = current.getNext()
        temp = node(item)
        previous.setNext(temp)

    def printList(self):
        current = self.head
        isEnd = False
        while not isEnd:
            if current.getNext() == None:
                print(current.getData())
                isEnd = True
            else:
                print(current.getData())
                current = current.getNext()

    def index(self, item):
        current = self.head
        count = 0
        found = False
        while current != None and not found:
            if current.getData() == item:
                found = True
            else:
                current = current.getNext()
                count = count + 1
        if found:
            return count
        else:
            return -1

    def pop(self, position):
        current = self.head
        previous = None
        count = 0
        while count < position:
            previous = current
            current = current.getNext()
            count = count + 1
        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())
        return current.getData()

class OrderedList:
    def __init__(self):
        self.head = None

    def search(self, item):
        current = self.head
        found = False
        stop = False
        while current != None and not found and not stop:
            if current.getData() == item:
                found = True
            else:
                if current.getData() > item:
                    stop = True
                else:
                    current = current.getNext()

        return found

    def add(self, item):
        current = self.head
        previous = None
        stop = False
        while current != None and not stop:
            if current.getData() > item:
                stop = True
            else:
                previous = current
                current = current.getNext()

        temp = node(item)
        if previous == None:
            temp.setNext(self.head)
            self.head = temp
        else:
            temp.setNext(current)
            previous.setNext(temp)

# build on this ordered list
############## Linked List Implementation End ##############

########## hashing ###########
def hash(astring, tablesize):
    sum = 0
    for pos in range(len(astring)):
        sum = sum + ord(astring[pos])

    return sum%tablesize

######### hashing class implmentation #######
class HashTable:
    def __init__(self):
        self.size = 11
        self.slots = [None] * self.size
        self.data = [None] * self.size

    def put(self,key,data):
      hashvalue = self.hashfunction(key,len(self.slots))

      if self.slots[hashvalue] == None:
        self.slots[hashvalue] = key
        self.data[hashvalue] = data
      else:
        if self.slots[hashvalue] == key:
          self.data[hashvalue] = data  #replace
        else:
          nextslot = self.rehash(hashvalue,len(self.slots))
          while self.slots[nextslot] != None and \
                          self.slots[nextslot] != key:
            nextslot = self.rehash(nextslot,len(self.slots))

          if self.slots[nextslot] == None:
            self.slots[nextslot]=key
            self.data[nextslot]=data
          else:
            self.data[nextslot] = data #replace

    def hashfunction(self,key,size):
         return key%size

    def rehash(self,oldhash,size):
        return (oldhash+1)%size

    def get(self, key):
        startslot = self.hashfunction(key, len(self.slots))

        data = None
        stop = False
        found = False
        position = startslot
        while self.slots[position] != None and \
                not found and not stop:
            if self.slots[position] == key:
                found = True
                data = self.data[position]
            else:
                position = self.rehash(position, len(self.slots))
                if position == startslot:
                    stop = True
        return data

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, data):
        self.put(key, data)
######### hashing class implmentation end#######


H=HashTable()
H[54]="cat"
H[26]="dog"
H[93]="lion"
H[17]="tiger"
H[77]="bird"
H[31]="cow"
H[44]="goat"
H[55]="pig"
H[20]="chicken"
H.slots
