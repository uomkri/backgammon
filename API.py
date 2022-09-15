from api.actions.db_actions import DBActions
from flask.json import jsonify
from flask import send_file
from api.actions.api_actions import ApiActions
from api.flask_init import app
from flask import request


from flask_cors import CORS, cross_origin

actions = ApiActions()
db_actions = DBActions()

@app.route("/start/<name>")
@cross_origin()
def start(name):
    pvp_string = request.args.get("is_pvp")
    pvp = False
    if pvp_string == "true":
        pvp = True
    return actions.start(name, pvp)
    
@app.route("/move/<uid>", methods=['POST'])
@cross_origin()
def move(uid):
    return actions.move(uid)

@app.route("/getDatabase")
@cross_origin()
def get_database():
    return jsonify(db_actions.get_database())

@app.route("/offerDoubling/<uid>", methods=['POST'])
@cross_origin()
def doubling_offered(uid):
    return actions.double_offered(uid)

@app.route("/kisa/<uid>", methods=['POST'])
@cross_origin()
def doubling_agreement(uid):
    answer = request.args.get("answer")
    return actions.doubling_agreement(uid, answer)

@app.route("/getWinEvaluations/<uid>")
@cross_origin()
def get_win_eval(uid):
    res = actions.get_win_eval(uid)
    return jsonify(res)

@app.route("/updateMultiplier/<uid>", methods=['POST'])
@cross_origin()
def update_multiplier(uid):
    m = request.json.multiplier
    actions.update_multiplier(uid, m)
    return jsonify("ok")

@app.route("/getLastModel/<num>", methods=['GET'])
@cross_origin()
def get_last_model(num):
    path="../new_nural_network/saved/DR.png"
    return send_file(path, as_attachment=True)

@app.route("/getWinProbabilityDistribution/<uid>", methods=['GET'])
@cross_origin()
def get_win_prob_dist(uid):
    return jsonify(actions.get_win_probs(uid))
