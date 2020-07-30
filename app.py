# ---- YOUR APP STARTS HERE ----
# -- Import section --
from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
from bson.objectid import ObjectId
import os
from flask_pymongo import PyMongo
from flask import session
import analysis
import bcrypt



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
    session['updates'] = ""
    session['creditriskanalysisres'] = -1
    return render_template("index.html")

@app.route('/userlogin')
def userlogin():
    session['logout'] = ""
    return render_template("userlog.html")

@app.route('/investorlogin')
def investorlogin():
    session['logout'] = ""
    return render_template("investorlog.html")

@app.route('/investorhomepage')
def investorhomepage():
    users_collection=mongo.db.users
    userdict= list(users_collection.find({}))
    
    investorinfo = mongo.db.investorinfo
    projectsinvestedin = list(investorinfo.find({'investor':session['username']}))
    numinvested = len(projectsinvestedin)
    proj_list = []
    total_money_spent = 0
    for item in projectsinvestedin:
        proj_list.append(item['project'])
        total_money_spent+= float(item['invested'])


    props = {'users':userdict,
            'num':numinvested,
            'projlist':proj_list,
            'total':total_money_spent}
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
            investors.insert({"username" : given_username,"password": str(bcrypt.hashpw(given_password.encode('utf-8'), bcrypt.gensalt()), "utf-8")})
            session['username'] = given_username
            return investorhomepage()
        else:
            pwfromdata = list(investors.find({'username':given_username}))[0]['password']
            if bcrypt.checkpw(given_password.encode("utf-8"), pwfromdata.encode("utf-8")):
                session['username'] = given_username
                return investorhomepage()
            else:
                session['error'] = "username already taken"
                return render_template('investorlog.html')

@app.route('/userhomepage')
def userhomepage():
    userinfo = mongo.db.userinfo
    money = list(userinfo.find({'username':session['username']}))[0]['invested']
    print(list(userinfo.find({'username':session['username']}))[0]['invested'])
    moneyreq = list(userinfo.find({'username':session['username']}))[0]['moneyreq']
    investorinfo = mongo.db.investorinfo
    investorsforcurruser = list(investorinfo.find({'project':session['username']}))

    progress_percent = 0.0
    if float(moneyreq) > 0.0:
        progress_percent = float(money)/float(moneyreq) * 100

    props = {
        "money":"${:,.2f}".format(money),
        "investorslist":investorsforcurruser,
        "numinvestors":len(investorsforcurruser),
        "moneyreq":"${:,.2f}".format(int(moneyreq)),
        "progress": progress_percent
    }
    return render_template("userhomepage.html",props=props)

@app.route('/userlogintomongo', methods = ['GET', 'POST'])
def userlogintomongo():
    users = mongo.db.users
    given_username = request.form['username']
    given_password = request.form['password']

    userinfo = mongo.db.userinfo

    if request.method == 'GET':
        return "you are getting some info"
    else:
        if not list(users.find({'username':given_username})):
            users.insert({"username" : given_username,"password": str(bcrypt.hashpw(given_password.encode('utf-8'), bcrypt.gensalt()), "utf-8")})
            userinfo.insert({"username" : given_username,"age": 0,"ed":0,"employ":0,"address":0,"income":0,"debtinc":0,"creddebt":0,"othdebt":0,"invested":0,"moneyreq":0})
            session['username'] = given_username
            return userhomepage()
        else:
            pwfromdata = list(users.find({'username':given_username}))[0]['password']
            if bcrypt.checkpw(given_password.encode("utf-8"), pwfromdata.encode("utf-8")):
                session['username'] = given_username
                return userhomepage()
            else:
                session['error'] = "username already taken"
                return render_template('userlog.html')

# @app.route('/user/<id>')
# def user(id):
#     users_collection = mongo.db.users
#     currUser= list(users_collection.find({'_id':ObjectId(id)}))
#     session['currentuser'] = currUser[0]['username']
#     return render_template("userview.html", curr_user=currUser[0])

@app.route('/updateinfo', methods = ['GET', 'POST'])
def updateinfo():
    if request.method == 'GET':
        return "you are getting some info"
    else:
        userinfo = mongo.db.userinfo
        username = session['username']
        age = request.form['age']
        ed = request.form['ed']
        employ = request.form['employ']
        address = request.form['address']
        income = request.form['income']
        debtinc = request.form['debtinc']
        creddebt = request.form['creddebt']
        othdebt = request.form['othdebt']
        moneyreq = request.form['moneyyouwant']


        if not list(userinfo.find({'username':username})):
            userinfo.insert({"username" : username,"age": age,"ed":ed,"employ":employ,"address":address,"income":income,"debtinc":debtinc,"creddebt":creddebt,"othdebt":othdebt,"invested":0,"moneyreq":moneyreq})
            session['updates'] = 'submitted'
        else:
            # idofuser = list(userinfo.find({'username':username})[0]['_id']
            userinfo.update(
                { 'username': username },
                { "$set":
                    {
                        "age": age,"ed":ed,"employ":employ,"address":address,"income":income,"debtinc":debtinc,"creddebt":creddebt,"othdebt":othdebt, "moneyreq":moneyreq
                    }
                }
                )
            session['updates'] = 'updated'
        return "<a href = '/userhomepage'>redirect</a>"
        #return redirect(url_for("userhomepage")) UNCOMMENT THIS WHEN GOING TO HEROKU

@app.route('/runanalysis', methods = ['GET', 'POST'])
def runanalysis():
    if request.method == 'GET':
        return "you are getting some info"
    else:
        user_for_val = request.form['users']
        session['currentuser'] = user_for_val
        userinfo = mongo.db.userinfo

        userls = list(userinfo.find({'username':user_for_val}))[0]

        age = userls['age']
        ed = userls['ed']
        employ = userls['employ']
        address = userls['address']
        income = userls['income']
        debtinc = userls['debtinc']
        creddebt = userls['creddebt']
        othdebt = userls['othdebt']

        users_collection=mongo.db.users
        userdict= list(users_collection.find({}))

        investorinfo = mongo.db.investorinfo
        projectsinvestedin = list(investorinfo.find({'investor':session['username']}))
        numinvested = len(projectsinvestedin)
        proj_list = []
        total_money_spent = 0
        for item in projectsinvestedin:
            proj_list.append(item['project'])
            total_money_spent+= float(item['invested'])

        res = analysis.predict(float(age),float(ed),float(employ),float(address),float(income),float(debtinc),float(creddebt),float(othdebt))
        session['creditriskanalysisres'] = res
        if int(res) == 0:
            props = {
                "result" : "will not default",
                "username":user_for_val,
                'users':userdict,
                'num':numinvested,
            'projlist':proj_list,
            'total':total_money_spent
            }
        else:
            props = {
                "result" : "will default",
                "username":user_for_val,
                'users':userdict,
                'num':numinvested,
            'projlist':proj_list,
            'total':total_money_spent
            }


        return render_template("investorhomepage.html", props=props)

@app.route('/invest', methods = ['GET', 'POST'])
def invest():
    if request.method == 'GET':
        return "you are getting some info"
    else:
        username = session['currentuser']
        userinfo = mongo.db.userinfo
        curr_money = list(userinfo.find({'username':username}))[0]['invested']
        new_money = float(curr_money) + float(request.form['investment'])
        userinfo.update(
                { 'username': username },
                { "$set":
                    {
                        "invested": new_money
                    }
                }
                )
        
        investorinfo = mongo.db.investorinfo #FIX THE MULTIPLE INVESTMENT THING
        if not list(userinfo.find({'project':username, 'investor':session['username']})):
            investorinfo.insert({"investor": session['username'], "project" : username,"invested":request.form['investment']})
        else:
            print(updating)
            currentinvestment = float(list(userinfo.find({'project':username, 'investor':session['username']}))[0]['invested'])
            total = currentinvestment + float(request.form['investment'])
            investorinfo.update(
                { 'project': username , 'investor': session['username']},
                { "$set":
                    {
                        "invested": total
                    }
                }
                )
        users_collection=mongo.db.users
        userdict= list(users_collection.find({}))

        projectsinvestedin = list(investorinfo.find({'investor':session['username']}))
        numinvested = len(projectsinvestedin)
        proj_list = []
        total_money_spent = 0
        for item in projectsinvestedin:
            proj_list.append(item['project'])
            total_money_spent+= float(item['invested'])

        props = {
                "result" : session['creditriskanalysisres'],
                "username":session['username'],
                'users':userdict,
                'input':f"you have invested ${str(request.form['investment'])}",
                'num':numinvested,
            'projlist':proj_list,
            'total':total_money_spent
                }

        return render_template("investorhomepage.html", props=props)

@app.route('/logout')
def logout():
    session.clear()
    session['logout'] = "You have been logged out."
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    session['logout'] = "404 error - page not found"
    
    return render_template('index.html'), 404

# testing block content just for funzies - ignore the test.html and base.html files in template for now
# documentation: https://jinja.palletsprojects.com/en/2.11.x/templates/#child-template
# @app.route('/test')
# def test():
#     return render_template("test.html")