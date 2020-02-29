# inspired by this:
# https://www.tutorialspoint.com/How-can-I-create-a-directory-if-it-does-not-exist-using-Python
import os
from pathlib import Path

dirs = ['results',
        'logs',
        'plots',
        'splits',
        'nn_output',
        'nn_logs',
        'nn_plots',
]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)
