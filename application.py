
from sqlalchemy import create_engine
import instance
from pandas.io import sql
from flask import Flask, jsonify, render_template, request

application = Flask(__name__)

database = instance.getDatabase()
engine = create_engine(database)

df = 0

@application.route('/')
def homepage():

    global df
    print str(df.get_values())
    return str(df.get_values())


if __name__ == '__main__':
    df = sql.read_frame("SELECT * FROM TBLelevationV1 LIMIT 10", engine)
    application.debug = False
    application.run(host='0.0.0.0', port=5000)