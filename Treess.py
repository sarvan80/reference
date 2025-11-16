# Functional implementation of a tree Pythonds
# https://realpython.com/python-magic-methods/ --Magic methods very imprtant
# https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python --print tree level by level (Tree visualization)

# https://www.pythontutorial.net/python-built-in-functions/python-next/
### linux for command line ###
# https://towardsdatascience.com/start-using-linux-commands-to-quick-analyze-structured-data-not-pandas-f63065842269
# https://towardsdatascience.com/start-using-linux-commands-to-quick-analyze-structured-data-not-pandas-f63065842269


# https://www.geeksforgeeks.org/difference-between-binary-tree-and-binary-search-tree/

# B+ tree or m-way search tree
# https://www.youtube.com/watch?v=K1a2Bk8NrYQ -- important

def BinaryTree(r):
    return [r, [], []]

def insertLeft(root,newBranch):
    t = root.pop(1)
    if len(t) > 1:
        root.insert(1,[newBranch,t,[]]) # this is the key line --most imp
    else:
        root.insert(1,[newBranch, [], []])
    return root

def insertRight(root,newBranch):
    t = root.pop(2)
    if len(t) > 1:
        root.insert(2,[newBranch,[],t])# key line
    else:
        root.insert(2,[newBranch,[],[]])
    return root

def getRootVal(root):
    return root[0]

def setRootVal(root,newVal):
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


class BinaryTree:
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode): # recursivw class object vs recursive function (Bintree vs Binsearch tree)
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)# this becomes and is assigned the root
        else: # this solves the going down the tree problem
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

# BinHeap Implementation
# Pythonds
# https://runestone.academy/runestone/books/published/pythonds/Trees/BinaryHeapImplementation.html
# heap is a very good data structure to implement a priority queue (added and imp)
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

# logic is easier if we understand access accross the tree upwards and downwards and  recursion and dp (added and very imp)
class TreeNode:
    def __init__(self, key, val, left=None, right=None, parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent
        self.balanceFactor = 0

    def __str__(self):
        return str(self.key) + ' : ' + str(self.payload) + ' : ' + str(self.leftChild) + ' : ' + str(self.rightChild)

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

    def replaceNodeData(self, key, value, lc, rc):# need to understand this
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self

    def findSuccessor(self):# need to understand this
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

    def spliceOut(self):# need to understand this
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

    def findMin(self):# this is a helper function of findSuccessor
        current = self
        while current.hasLeftChild():
            current = current.leftChild
        return current

    # https: // stackoverflow.com / questions / 46532735 / how - an - iterable - object - is -iterated - without - next (--very important)
    def __iter__(self):#  at first glance you might think that the code is not recursive. However, remember that __iter__ overrides the for x in operation for iteration, so it really is recursive! Because it is recursive over TreeNode instances the __iter__ method is defined in the TreeNode class. (when does iter get fired)
        """The standard inorder traversal of a binary tree."""
        if self:
            if self.hasLeftChild():
                for elem in self.leftChild:
                    yield elem
            yield self.key
            if self.hasRightChild():
                for elem in self.rightChild:
                    yield elem


# Binary search tree implementation Pythonds
import unittest


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
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)# key recursive method
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

    def inorder(self):# at first glance you might think that the code is not recursive. However, remember that __iter__ overrides the for x in operation for iteration, so it really is recursive! Because it is recursive over TreeNode instances the __iter__ method is defined in the TreeNode class.
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
#

import unittest
# from .bst import BinarySearchTree,TreeNode

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
'''
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

# list representation
myTree = ['a',   #root
      ['b',  #left subtree
       ['d', [], []],
       ['e', [], []] ],
      ['c',  #right subtree
       ['f', [], []],
       [] ]
     ]

print(myTree)
print('left subtree = ', myTree[1])

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

'''
'''
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

'''


t = BinaryTree(7)#left side's right side and right side's left side are always empty starting from the root --very imp
t.insertLeft(3)
t.insertRight(9)

t.insertLeft(4)
t.insertRight(5)

t.insertLeft(7)
t.insertRight(19)

t.insertLeft(31)
t.insertRight(8)

inorder(t)

### most important to understand iter ###
mytree = BinarySearchTree()
mytree[3]="red"
mytree[411]="blue"
mytree[6]="yellow"
mytree[2]="at"

mytree[13]="red1"
mytree[41]="blue1"
mytree[16]="yellow1"
mytree[21]="at1"


print(mytree[6])
print(mytree[2])

for i in mytree: # this is where the iterator actually kicks in which was explained b4
    print(i)

### most important to understand iter end ###

bst = AVLTree()
'''
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
'''
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

#### try this later but immediately ########
'''
assert bst.root.key == 45
assert bst.root.leftChild.key == 40
assert bst.root.rightChild.key == 50
assert bst.root.balanceFactor == 0
assert bst.root.leftChild.balanceFactor == 0
assert bst.root.rightChild.balanceFactor == -1
'''
#### try this later but immediately end ########
# Local Variables:
# py-which-shell: "python3"
# End:

def printTree(node, level=0):
    if node != None:
        printTree(node.leftChild, level + 1)
        print(' ' * 4 * level + '-> ' + str(node.key) + str(node.payload))
        printTree(node.rightChild, level + 1)

def printTree1(node, level=0):
    if node != None:
        printTree1(node.leftChild, level + 1)
        print(' ' * 4 * level + '-> ' + str(node.key))
        printTree1(node.rightChild, level + 1)

print("keep typing irrespective of the suggestion")

# printTree(mytree.root) # check with .root
printTree(bst.root) # check with .root
print('Tree printed')

# c
# Key aspects--Magic methods overloading in objects/(python stdby,copy & past--total 25 hrs with steps 2//replicate)




