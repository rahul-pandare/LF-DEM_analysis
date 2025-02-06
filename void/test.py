import os
import matplotlib                   # type: ignore
import numpy             as     np  # type: ignore
import matplotlib.pyplot as     plt # type: ignore
import random

def outer_function():
    inner_function()

def inner_function():
    print("Hello from the inner function!")

outer_function()