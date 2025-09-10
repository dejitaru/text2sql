from flask import Flask, request, render_template_string
import pickle
import pandas as pd
from pathlib import Path

app_ui = Flask(__name__)

# Reusar la misma instancia de LocalVanna
from app import vn

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Entrenamiento Vanna</title>
</head>
<body>
<h2>Agregar Patrón de Entrenamiento</h2>
<form method="POST">
    Pregunta: <input type="text" name="question" style="width:500px;"><br><br>
    SQL: <textarea name="sql" style="width:500px; height:100px;"></textarea><br><br>
    Documentación: <textarea name="documentation" style="width:500px; height:50px;"></textarea><br><br>
    <button type="submit">Agregar</button>
</form>

<h3>Datos de entrenamiento actuales:</h3>
<pre>{{training_data}}</pre>
</body>
</html>
"""

@app_ui.route('/', methods=['GET', 'POST'])
def add_pattern():
    if request.method == 'POST':
        question = request.form.get('question')
        sql = request.form.get('sql')
        documentation = request.form.get('documentation')
        
        # Agregar al DataFrame y guardar
        vn._training_data = pd.concat([
            vn._training_data,
            pd.DataFrame([{
                "question": question,
                "sql": sql,
                "ddl": None,
                "documentation": documentation
            }])
        ], ignore_index=True)
        
        # Guardar en pickle
        vn.training_file.parent.mkdir(exist_ok=True)
        with open(vn.training_file, 'wb') as f:
            pickle.dump(vn._training_data, f)
    
    return render_template_string(HTML_FORM, training_data=vn._training_data.to_string())

if __name__ == '__main__':
    app_ui.run(port=5002, debug=True)
