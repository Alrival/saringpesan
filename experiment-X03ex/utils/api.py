from quart import Quart, request, Response
import json
import re

from utils.evaluate import predict_string

# from twilio.rest import Client
# from twilio.twiml.messaging_response import MessagingResponse

# auth_token = '75bbfc9eddc4ffbf01e634aec3e42964'
# account_sid = 'AC441def446c770efbe657ef4cbd53fc4c'
server_addr = 'localhost'
server_port = 4567

app = Quart(__name__)
app_model = None
app_tokenizer = None
app_i2w = None
app_w2i = None

# @app.route("/wa", methods=['POST'])
# async def wa_reply():
#     """Send a dynamic reply to an incoming text message"""
#     # Get the message the user sent our Twilio number
#     incoming_msg = request.values.get('Body', None)

#     # Start our TwiML response
#     response = MessagingResponse()
#     msg = response.message()
    
#     check_str = predict_string(incoming_msg, app_i2w, app_model, app_tokenizer)
#     text = '\n'.join(('Message:',
#                       '\"\"\"',
#                       '{}'.format(check_str['sentence']),
#                       '\"\"\"',
#                       'Result:',
#                       '{} ({})'.format(check_str['label'], check_str['prediction']) ))
#     msg.body(text)

#     return str(response)

@app.route("/android/api/single", methods=['POST'])
async def singleJSONData():
    data = await request.get_json()    
    result_data = {}

    for key,val in data.items():
        predict = predict_string(val, app_i2w, app_model, app_tokenizer)
        result_data.update({"flag": "{}#{}".format(app_w2i[ predict['label'] ],predict['prediction'])})
        break

    response = Response(response=json.dumps(result_data),
                        status=200,
                        mimetype="application/json")
    return response

@app.route("/android/api/bulk", methods=['POST'])
async def bulkJSONData():
    bulk_data = await request.get_json()    
    result_data = {}

    for key,val in bulk_data.items():
        msg_id = re.search('\d+', key).group(0)
        predict = predict_string(val, app_i2w, app_model, app_tokenizer)
        result_data.update({"flag"+str(msg_id): "{}#{}".format(app_w2i[ predict['label'] ],predict['prediction'])})

    response = Response(response=json.dumps(result_data),
                        status=200,
                        mimetype="application/json")
    return response

def init_server(model=None, tokenizer=None, i2w=None, w2i=None):
    global app_model, app_tokenizer, app_i2w, app_w2i
    app_model = model
    app_tokenizer = tokenizer
    app_i2w = i2w
    app_w2i = w2i
    
    app.run(host=server_addr, port=server_port)