#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:53:10 2020

@author: deviantpadam
"""

from flask import Flask, jsonify, request, render_template
from torch2vec.torch2vec import LoadModel

app = Flask(__name__)
model = LoadModel('model/weights.npy')

@app.route('/')
def home():
    doc_ids = model.embeddings[:,0].astype('int')
    if not request.args.get('num'):
        num = 0
    else: 
        num = int(request.args.get('num'))
        
    return render_template('home.html', doc_ids = doc_ids[25*num:25*(num+1)], num=num)

@app.route('/torch')
def torch():
    id = request.args.get('id')
    sim = model.similar_docs(int(id),topk=10,use='torch') #this will be slow 
    #will try to set a limit
    similar = {'ids':sim[0],'sim_score':sim[1]}
    return jsonify(similar) #'I think have to add a limit or something to make it efficient it is not usable in websites'

@app.route('/sklearn')
def sklearn():
    id = request.args.get('id')
    sim = model.similar_docs(int(id),topk=10,use='sklearn') #this is faster
    #results may be different i dont know why?
    sim = [int(i) for i in sim]
    similar = {'ids':sim}
    return jsonify(similar)


@app.route('/select')
def select():
    ids = request.args.get('id')
    return render_template('select.html',ids=ids)

if "__main__"==__name__:
    app.run()
    
 