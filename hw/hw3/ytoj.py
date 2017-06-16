#!/usr/bin/env python


import yaml
import json
import sys


with open(sys.argv[1]) as file_:
    print json.dumps(yaml.load(file_), indent=2)
