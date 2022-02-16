"""
This approach allows for clean relative imports when dealing with Juypter
notebooks. All relative imports happen within this file and note in
Jupyter notebooks.
See: https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im
"""

import os
import sys

module_path = os.path.dirname(os.getcwd()) + '/scripts'
if module_path not in sys.path:
    sys.path.append(module_path)

os.chdir("../..")
