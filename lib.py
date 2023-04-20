import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import yaml
import streamlit as st
import sys
import os
import platform
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression