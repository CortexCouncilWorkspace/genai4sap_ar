from functools import wraps
from flask import Flask, jsonify, Response, request, redirect, url_for
import flask
import os
import sys
import configparser
from cache import MemoryCache
from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat
from waitress import serve
import json

# Loading Configuration Values
# Loading Configuration Values
module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)
config = configparser.ConfigParser()
config.read(module_path+'/config.ini')

PROJECT_ID = config['CONFIG']['PROJECT_ID']
DATASET_ID = config['CONFIG']['DATASET_ID'] 
REGION_ID = config['CONFIG']['REGION_ID'] 
MODEL=config['CONFIG']['MODEL']
LANGUAGE=config['CONFIG']['LANGUAGE']

#Loading API Key
# api_key_file = os.open('/secrets/api_key', 'os.O_RDONLY ', '0o666')
# API_KEY = json.load(api_key_file)
with open("auth/api_key", "r" ) as api_key:
    API_KEY = api_key.readlines()[0]

# SETUP
cache = MemoryCache()

class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
    def __init__(self, config={'path':'/chroma_data'}):
        ChromaDB_VectorStore.__init__(self, config=config)
        GoogleGeminiChat.__init__(self, config={'api_key': f'{API_KEY}', 'model': 'gemini-1.5-pro-001', 'language':'Spanish'})

vn = MyVanna()

vn.connect_to_bigquery(project_id=PROJECT_ID)

app = Flask(__name__, static_url_path='')

def requires_cache(fields):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            id = request.args.get('id')

            if id is None:
                return jsonify({"type": "error", "error": "No id provided"})
            
            for field in fields:
                if cache.get(id=id, field=field) is None:
                    return jsonify({"type": "error", "error": f"No {field} found"})
            
            field_values = {field: cache.get(id=id, field=field) for field in fields}
            
            # Add the id to the field_values
            field_values['id'] = id

            return f(*args, **field_values, **kwargs)
        return decorated
    return decorator

@app.route('/api/v0/generate_questions', methods=['GET'])
def generate_questions():
    return jsonify({
        "type": "question_list", 
        "questions": vn.generate_questions(),
        "header": "Aquí hay algunas preguntas sugeridas que puede hacer:"
        })

@app.route('/api/v0/generate_sql', methods=['GET'])
def generate_sql():
    question = flask.request.args.get('question')

    if question is None:
        return jsonify({"type": "error", "error": "No question provided"})

    id = cache.generate_id(question=question)
    sql = vn.generate_sql(question=question)

    cache.set(id=id, field='question', value=question)
    cache.set(id=id, field='sql', value=sql)

    return jsonify(
        {
            "type": "sql", 
            "id": id,
            "text": sql,
        })

@app.route('/api/v0/run_sql', methods=['GET'])
@requires_cache(['sql'])
def run_sql(id: str, sql: str):
    try:
        df = vn.run_sql(sql=sql)

        cache.set(id=id, field='df', value=df)

        return jsonify(
            {
                "type": "df", 
                "id": id,
                "df": df.head(10).to_json(orient='records',date_format='iso'),
            })

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/download_csv', methods=['GET'])
@requires_cache(['df'])
def download_csv(id: str, df):
    csv = df.to_csv()

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 f"attachment; filename={id}.csv"})

@app.route('/api/v0/generate_plotly_figure', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_plotly_figure(id: str, df, question, sql):
    try:
        code = vn.generate_plotly_code(question=question, sql=sql, df_metadata=f"Running df.dtypes gives:\n {df.dtypes}")
        fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
        fig_json = fig.to_json()

        cache.set(id=id, field='fig_json', value=fig_json)

        return jsonify(
            {
                "type": "plotly_figure", 
                "id": id,
                "fig": fig_json,
            })
    except Exception as e:
        # Print the stack trace
        import traceback
        traceback.print_exc()

        return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/get_training_data', methods=['GET'])
def get_training_data():
    df = vn.get_training_data()

    return jsonify(
    {
        "type": "df", 
        "id": "training_data",
        "df": df.head(25).to_json(orient='records'),
    })

@app.route('/api/v0/remove_training_data', methods=['POST'])
def remove_training_data():
    # Get id from the JSON body
    id = flask.request.json.get('id')

    if id is None:
        return jsonify({"type": "error", "error": "No id provided"})

    if vn.remove_training_data(id=id):
        return jsonify({"success": True})
    else:
        return jsonify({"type": "error", "error": "Couldn't remove training data"})

@app.route('/api/v0/train', methods=['POST'])
def add_training_data():
    question = flask.request.json.get('question')
    sql = flask.request.json.get('sql')
    ddl = flask.request.json.get('ddl')
    documentation = flask.request.json.get('documentation')

    try:
        id = vn.train(question=question, sql=sql, ddl=ddl, documentation=documentation)

        return jsonify({"id": id})
    except Exception as e:
        print("TRAINING ERROR", e)
        return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/generate_followup_questions', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_followup_questions(id: str, df, question, sql):
    followup_questions = vn.generate_followup_questions(question=question, sql=sql, df=df)

    cache.set(id=id, field='followup_questions', value=followup_questions)

    return jsonify(
        {
            "type": "question_list", 
            "id": id,
            "questions": followup_questions,
            "header": "Aqui estão algumas sugestões de perguntas que você pode fazer:"
        })

@app.route('/api/v0/load_question', methods=['GET'])
@requires_cache(['question', 'sql', 'df', 'fig_json', 'followup_questions'])
def load_question(id: str, question, sql, df, fig_json, followup_questions):
    try:
        return jsonify(
            {
                "type": "question_cache", 
                "id": id,
                "question": question,
                "sql": sql,
                "df": df.head(10).to_json(orient='records'),
                "fig": fig_json,
                "followup_questions": followup_questions,
            })

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/get_question_history', methods=['GET'])
def get_question_history():
    return jsonify({"type": "question_history", "questions": cache.get_all(field_list=['question']) })

@app.route('/api/v0/setup_train', methods=['POST'])
def setup_train():
    project_id_api = flask.request.json.get('project_id')
    dataset_id_api = flask.request.json.get('dataset_id')
    table_list_api = flask.request.json.get('table_list')
    if project_id_api and dataset_id_api is None:
        return jsonify({"type": "error", "error": "No project/dataset provided"})
    if project_id_api and dataset_id_api is not None:
        schema_query = f"""SELECT table_catalog, table_schema, table_name, column_name, data_type, description as `comment` FROM `{project_id_api}.{dataset_id_api}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS` WHERE table_name in ({table_list_api})"""
        df_information_schema = vn.run_sql(schema_query)
        plan = vn.get_training_plan_generic(df_information_schema)
        plan
        vn.train(plan=plan)
        return jsonify({"success": 'Training session finished successfully'})
    else:
        return jsonify({"type": "error", "error": "Error on training session!"})

@app.route('/')
def root():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)
