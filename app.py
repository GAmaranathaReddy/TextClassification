a# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:24:26 2019

@author: C5278763
"""

from flask import Flask
from pymongo import MongoClient
from bson import json_util
import json



MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DB_NAME = 'mitdev'
COLLECTION_NAME = 'team'


app= Flask(__name__)

@app.route('/')
def index():
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DB_NAME][COLLECTION_NAME]
    projects = collection.find()
    json_projects=[]
    for project in projects:                                            
        json_projects.append(project)
    json_projects=json.dumps(json_projects,default=json_util.default)
    connection.close()
    return json_projects
  
if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=True)