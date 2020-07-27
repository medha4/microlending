# ---- YOUR APP STARTS HERE ----
# -- Import section --
from flask import Flask
from flask import render_template
from flask import request
from bson.objectid import ObjectId
import os
from flask_pymongo import PyMongo
from flask import session

# -- Initialization section --
app = Flask(__name__)

app.secret_key = "loans"

# name of database
app.config['MONGO_DBNAME'] = 'microlending'

MONGO_USER = os.environ['user']
MONGO_PASSWORD = os.environ['password']

# URI of database
app.config['MONGO_URI'] = f'mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@cluster0.4lc7f.mongodb.net/microlending?retryWrites=true&w=majority'



mongo = PyMongo(app)

# -- Routes section --
@app.route('/')
@app.route('/index')
def index():
    session['error'] = ""
    return render_template("index.html")

@app.route('/userlogin')
def userlogin():
    return render_template("userlog.html")

@app.route('/investorlogin')
def investorlogin():
    return render_template("investorlog.html")

@app.route('/investorhomepage')
def investorhomepage():
    users_collection=mongo.db.users
    userdict= list(users_collection.find({}))
    props = {'users':userdict}
    return render_template("investorhomepage.html", props=props)

@app.route('/investlogintomongo', methods = ['GET', 'POST'])
def investlogintomongo():
    investors = mongo.db.investors
    given_username = request.form['username']
    #given_password = bcrypt.hashpw(request.form['password'].encode("utf-8"),bcrypt.gensalt())
    given_password = request.form['password']
    print(given_password)
    if request.method == 'GET':
        return "you are getting some info"
    else:
        if not list(investors.find({'username':given_username})):
            #investors.insert({"username" : given_username,"password": str(given_password,'utf-8')})
            investors.insert({"username" : given_username,"password": given_password})
            session['username'] = given_username
            return investorhomepage()
        else:
            #if list(investors.find({'username':given_username}))[0]['password'] == str(given_password,'utf-8'):
            if list(investors.find({'username':given_username}))[0]['password'] == given_password:
                session['username'] = given_username
                return investorhomepage()
            else:
                session['error'] = "username already taken"
                return render_template('investorlog.html')

@app.route('/userhomepage')
def userhomepage():
    return render_template("userhomepage.html")

@app.route('/userlogintomongo', methods = ['GET', 'POST'])
def userlogintomongo():
    users = mongo.db.users
    given_username = request.form['username']
    #given_password = bcrypt.hashpw(request.form['password'].encode("utf-8"),bcrypt.gensalt())
    given_password = request.form['password']
    print(given_password)
    if request.method == 'GET':
        return "you are getting some info"
    else:
        if not list(users.find({'username':given_username})):
            #users.insert({"username" : given_username,"password": str(given_password,'utf-8')})
            users.insert({"username" : given_username,"password": given_password})
            session['username'] = given_username
            return userhomepage()
        else:
            #if list(users.find({'username':given_username}))[0]['password'] == str(given_password,'utf-8'):
            if list(users.find({'username':given_username}))[0]['password'] == given_password:
                session['username'] = given_username
                return userhomepage()
            else:
                session['error'] = "username already taken"
                return render_template('userlog.html')

@app.route('/user/<id>')
def user(id):
    users_collection = mongo.db.users
    currUser= list(users_collection.find({'_id':ObjectId(id)}))

    return render_template("userview.html", curr_user=currUser[0])