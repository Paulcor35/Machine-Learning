# Source: Jacek Generowicz

############### Python is interactive ##################################

# You can interact with a running Python program and define and use
# new functions and classes interactively. This is extremely useful
# for experimenting and gaining an understanding of how your program
# works.

# Try
#
#     python 
# or
#     python -i program.py
#
# This should give you the Python prompt '>>>' which will run any
# python code you type.

############### Python is dynamically typed ############################

# Type checking is performed at run-time rather than compile
# time.

# Objects know their own type; variables do not know the type of
# object they contain.

# The type of the data stored in a variable may change at
# run-time. There are no type declarations.


# We use the variable 'a' without declaration.
a = 1
print(type(1))

# We can change the type of object stored in 'a' to a completely
# unrelated type.
a = 'hello'
print(type(a))

############### Block structure #######################################

# Python determines block structure by indentation.
if True:
    print("line one")   # First line of consequent block
    print("line two")
    print("line three") # Last line of consequent block
else:
    print("line four")  # First line of alternative block
    print("line five")
    print("line six")
    print("line seven") # Last line of alternative block

############### Python functions #######################################

# Python functions are defined using the 'def' keyword.
def add3(a,b,c):
    return a + b + c

# Python functions are called (invoked) just like C++ or Java
# functions
print(add3(1,2,3))
# Note that Python functions are polymorphic: they can operate on many
# different types
print(add3("this", " and ", "that"))

############## Loops ###################################################

# For-loops iterate over containers
for element in "hello":
    print(element)

for element in [1,2,3,4,5,10,100]:
    print(element)

# While loops are obvious
count = 0
while count < 10:
    print(count)
    count += 1
print("done")

############## Lists & Dictionaries  ###################################

# Lists are a versatile Python container type. They are heterogeneous.

lst = [1,'banana',[]]
print(type(lst))

# Subscripting works like it does in C++ arrays or vectors
for i in range(len(lst)):
    print(lst[i])

# Dictionaries are hash-tables: they map keys to values
d = {"one":1, "two":2}
print(type(d))
# What is the value associated with the key "one"?
print(d["one"])
# How can I add in extra key-value pairs into the dictionary?
d["three"] = 3
print(d)


############## Classes #################################################

# Use the 'class' keyword to create classes
class Base:

    # The constructor is called __init__
    def __init__(self, arg): 
        # The equivalent of C++/Java's 'this' is always an explicit
        # parameter, and is called 'self'
        
        # Rather than declaring data members, they are simply stuck
        # onto the instance, dynamically, at run-time.
        self.arg = arg
        # So, the data members you would declare in a class in C++ or
        # Java are (usually) stuck onto self in the constructor, in
        # Python.

    # Methods (member-functions in C++) are simply functions defined
    # in the class.
    def meth(self, arg): # Don't forget self !
        return self.arg + arg


# You can instantiate a class by calling it
b = Base(3)

# Method invocation looks just like it does in C++ and Java.
print(b.meth(4))

# Inheritance
class Derived(Base):

    def __init__(self, arg1, arg2):
        # Superclass constructors must be called explicitly, if you
        # want them to run.
        super().__init__(arg1)
        # or: Base.__init__(self, arg1)
        self.arg2 = arg2

    def meth2(self, arg):
        return self.arg + self.arg2 + arg


d = Derived(2,3)
# Derived has inherited meth()
print(d.meth(4))
# Derived has enlarged the interface it inherited from Base, by adding
# meth2()
d.meth2(2)



############## Exception handling  ###################################

try:
    1/0
except ZeroDivisionError:
    print("Handling ZeroDivisionError")
