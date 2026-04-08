import sys
import os
sys.path.append(os.getcwd())
from model.classifier import BirdClassifier

try:
    c = BirdClassifier()
    desc = c.get_bird_description("American Robin")
    print("SUCCESS")
    print(desc)
except Exception as e:
    print("FAILURE")
    print(e)
