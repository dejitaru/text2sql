from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import requests
import pickle
from pathlib import Path
from flask import Flask, jsonify, request, Response
from functools import wraps
from cache import MemoryCache

app = Flask(__name__)
cache = MemoryCache()

print("🚀 Iniciando Vanna Flask Server LOCAL...")

class LocalVanna:
    """Implementación local de Vanna que usa solo Ollama y almacenamiento local"""
    
    def __init__(self, model_name="llama3.2:latest", storage_path="./vanna_local_storage"):
        self.ollama_model = model_name
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Archivos de almacenamiento
        self.training_file = self.storage_path / "training_data.pkl"
        
        # Inicializar training data
        self._training_data = self._load_training_data()
        
        # Conexión MySQL
        self.mysql_connection = None
        
    def _load_training_data(self):
        """Carga los datos de entrenamiento desde archivo local"""
        if self.training_file.exists():
            try:
                with open(self.training_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Cargados {len(data)} registros de entrenamiento")
                return data
            except Exception as e:
                print(f"⚠️  Error cargando training data: {e}")
        
        print("⚠️  No se encontró archivo de entrenamiento")
        return pd.DataFrame(columns=["question", "sql", "ddl", "documentation"])
    
    def connect_mysql(self):
        """Conecta a MySQL usando PyMySQL"""
        import pymysql
        
        self.mysql_connection = pymysql.connect(
            host=os.environ.get('MYSQL_HOST', 'localhost'),
            port=int(os.environ.get('MYSQL_PORT', 3306)),
            user=os.environ.get('MYSQL_USER', 'root'),
            password=os.environ.get('MYSQL_PASSWORD', ''),
            database=os.environ.get('MYSQL_DATABASE', 'test'),
            charset='utf8mb4'
        )
    
    def run_sql(self, sql: str):
        """Ejecuta SQL con PyMySQL"""
        try:
            if not self.mysql_connection or not self.mysql_connection.open:
                self.connect_mysql()
            
            cursor = self.mysql_connection.cursor()
            cursor.execute(sql)
            
            # Para queries que devuelven datos
            if sql.strip().upper().startswith(('SELECT', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN')):
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                cursor.close()
                return pd.DataFrame(results, columns=columns)
            else:
                # Para queries que modifican datos
                self.mysql_connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                return pd.DataFrame([{"affected_rows": affected_rows}])
                
        except Exception as e:
            print(f"❌ Error ejecutando SQL '{sql[:50]}...': {e}")
            raise
    
    def get_training_data(self):
        """Retorna los datos de entrenamiento"""
        return self._training_data
    
    def submit_prompt(self, prompt: str, **kwargs):
        """Envía prompt a Ollama local"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60  # Aumentar timeout para queries complejas
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"❌ Error conectando con Ollama: {e}")
            return f"Error: No se pudo conectar con Ollama - {str(e)}"
    
    def generate_sql(self, question: str):
        """Genera SQL basado en la pregunta y datos de entrenamiento"""
        
        if len(self._training_data) == 0:
            raise Exception("No training data available. Please run train_vanna.py first")
        
        # Construir contexto desde datos de entrenamiento
        context_parts = []
        
        # Añadir DDLs (esquemas de tablas)
        ddls = self._training_data[self._training_data['ddl'].notna()]['ddl'].tolist()
        if ddls:
            context_parts.append("Database Schema:")
            for ddl in ddls[:3]:  # Limitar para no sobrecargar el prompt
                context_parts.append(ddl.strip())
        
        # Añadir ejemplos similares
        examples = self._training_data[
            (self._training_data['question'].notna()) & 
            (self._training_data['sql'].notna())
        ][['question', 'sql']]
        
        if len(examples) > 0:
            context_parts.append("\nExample Questions and SQL:")
            # Buscar ejemplos más relevantes (contienen palabras clave similares)
            question_lower = question.lower()
            relevant_examples = []
            
            for _, row in examples.iterrows():
                if any(word in row['question'].lower() for word in question_lower.split()):
                    relevant_examples.append(row)
            
            # Si no hay ejemplos relevantes, usar algunos aleatorios
            if not relevant_examples:
                relevant_examples = examples.head(5).to_dict('records')
            
            for example in relevant_examples[:5]:  # Máximo 5 ejemplos
                context_parts.append(f"Q: {example['question']}")
                context_parts.append(f"SQL: {example['sql']}")
        
        # Construir prompt
        context = "\n".join(context_parts)
        
        prompt = f"""You are a MySQL SQL expert. Given the database schema and examples below, generate a SQL query to answer the user's question.

{context}

User Question: {question}

Requirements:
- Generate ONLY the SQL query, no explanations
- Use valid MySQL syntax
- End the query with semicolon (;)
- If asking for "all" data, use SELECT * 
- If counting, use COUNT(*)
- Use LIMIT if appropriate

SQL Query:"""

        # Obtener respuesta de Ollama
        response = self.submit_prompt(prompt)
        
        # Limpiar respuesta
        sql = self._clean_sql_response(response)
        return sql
    
    def _clean_sql_response(self, response: str) -> str:
        """Limpia la respuesta de Ollama para extraer solo el SQL"""
        lines = response.strip().split('\n')
        
        # Buscar líneas que parezcan SQL
        sql_lines = []
        in_code_block = False
        
        for line in lines:
            line = line.strip()
            
            # Manejar bloques de código
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if not line:
                continue
            
            # Si estamos en un bloque de código, añadir todo
            if in_code_block:
                sql_lines.append(line)
                continue
                
            # Buscar líneas que empiecen con palabras clave SQL
            if any(line.upper().startswith(keyword) for keyword in 
                   ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'SHOW', 'DESCRIBE', 'DESC', 'CREATE', 'ALTER', 'DROP', 'WITH']):
                sql_lines.append(line)
            elif sql_lines and not any(line.startswith(marker) for marker in ['--', '#', '//', 'Note:', 'Explanation:']):
                # Si ya tenemos SQL y la línea no es comentario, podría ser continuación
                sql_lines.append(line)
        
        if sql_lines:
            sql = ' '.join(sql_lines)
            # Limpiar caracteres extra
            sql = sql.replace('```sql', '').replace('```', '').strip()
            # Asegurar que termine con ;
            if not sql.endswith(';'):
                sql += ';'
            return sql
        
        # Si no encontramos SQL claro, buscar en la respuesta original
        response_clean = response.strip()
        if any(keyword in response_clean.upper() for keyword in ['SELECT', 'SHOW', 'DESCRIBE']):
            if not response_clean.endswith(';'):
                response_clean += ';'
            return response_clean
        
        return response_clean

# Crear instancia de Vanna local
vn = LocalVanna(
    model_name=os.environ.get("OLLAMA_MODEL", "llama3.2:latest"),
    storage_path="./vanna_local_storage"
)

# Conectar a MySQL
try:
    vn.connect_mysql()
    print("✅ Conectado a MySQL")
except Exception as e:
    print(f"❌ Error conectando a MySQL: {e}")
    print("💡 Asegúrate de que MySQL esté ejecutándose y las credenciales sean correctas")
    exit(1)

# --- DECORADOR PARA CACHE ---
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
            field_values['id'] = id
            return f(**field_values, **kwargs)
        return decorated
    return decorator

# --- ENDPOINTS ---
@app.route('/api/v0/generate_sql', methods=['GET'])
def generate_sql():
    question = request.args.get('question')
    if not question:
        return jsonify({"type": "error", "error": "No question provided"})
    
    id = cache.generate_id(question=question)
    
    try:
        print(f"🤖 Generando SQL para: {question}")
        
        # Verificar que tenemos datos de entrenamiento
        training_data = vn.get_training_data()
        if training_data is None or len(training_data) == 0:
            return jsonify({
                "type": "error", 
                "error": "No training data found. Please run 'python train_vanna.py' first"
            })
        
        sql = vn.generate_sql(question=question)
        print(f"✅ SQL generado: {sql[:100]}...")
        
    except Exception as e:
        print(f"❌ Error generando SQL: {e}")
        error_msg = str(e)
        if "No training data" in error_msg:
            error_msg = "No training data found. Please run: python train_vanna.py"
        sql = f"-- Error: {error_msg}"
    
    cache.set(id=id, field='question', value=question)
    cache.set(id=id, field='sql', value=sql)
    return jsonify({"type": "sql", "id": id, "text": sql})

@app.route('/api/v0/run_sql', methods=['GET'])
@requires_cache(['sql'])
def run_sql(id: str, sql: str):
    try:
        # Verificar que no sea un error
        if sql.startswith('-- Error:'):
            return jsonify({"type": "error", "error": "Cannot execute SQL with errors"})
        
        print(f"🔍 Ejecutando SQL: {sql[:100]}...")
        df = vn.run_sql(sql=sql)
        cache.set(id=id, field='df', value=df)
        return jsonify({
            "type": "df", 
            "id": id, 
            "df": df.head(20).to_json(orient='records'),  # Mostrar más filas
            "total_rows": len(df)
        })
    except Exception as e:
        print(f"❌ Error ejecutando SQL: {e}")
        return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/download_csv', methods=['GET'])
@requires_cache(['df'])
def download_csv(id: str, df):
    csv = df.to_csv(index=False)
    return Response(
        csv, 
        mimetype="text/csv", 
        headers={"Content-disposition": f"attachment; filename=query_{id}.csv"}
    )

@app.route('/api/v0/test_connection', methods=['GET'])
def test_connection():
    """Endpoint para probar las conexiones"""
    results = {}
    
    # Test MySQL
    try:
        test_df = vn.run_sql("SELECT 1 as test, NOW() as current_time")
        results['mysql'] = f"✅ Conectado - {len(test_df)} filas"
    except Exception as e:
        results['mysql'] = f"❌ Error: {str(e)}"
    
    # Test Ollama
    try:
        test_response = vn.submit_prompt("Say 'Hello from Ollama' in one line")
        results['ollama'] = f"✅ Respuesta: {test_response[:50]}..."
    except Exception as e:
        results['ollama'] = f"❌ Error: {str(e)}"
    
    # Test Training Data
    try:
        training_data = vn.get_training_data()
        training_count = len(training_data) if training_data is not None else 0
        if training_count > 0:
            # Contar tipos de datos
            ddl_count = len(training_data[training_data['ddl'].notna()])
            question_count = len(training_data[training_data['question'].notna()])
            results['training'] = f"✅ {training_count} registros (DDL:{ddl_count}, Q:{question_count})"
        else:
            results['training'] = "⚠️  Sin datos de entrenamiento - ejecuta train_vanna.py"
    except Exception as e:
        results['training'] = f"❌ Error: {str(e)}"
    
    return jsonify(results)

@app.route('/api/v0/tables', methods=['GET'])
def get_tables():
    """Endpoint para listar todas las tablas"""
    try:
        tables_df = vn.run_sql("SHOW TABLES")
        table_column = tables_df.columns[0]
        tables = tables_df[table_column].tolist()
        
        # Información de cada tabla
        tables_info = []
        for table in tables:
            try:
                # Obtener info básica
                count_df = vn.run_sql(f"SELECT COUNT(*) as row_count FROM {table}")
                describe_df = vn.run_sql(f"DESCRIBE {table}")
                
                tables_info.append({
                    "name": table,
                    "row_count": count_df.iloc[0]['row_count'],
                    "column_count": len(describe_df),
                    "columns": describe_df['Field'].tolist()
                })
            except Exception as e:
                tables_info.append({
                    "name": table,
                    "error": str(e)
                })
        
        return jsonify({"type": "tables", "count": len(tables), "tables": tables_info})
    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/training_data', methods=['GET'])
def get_training_data():
    """Endpoint para ver estadísticas de datos de entrenamiento"""
    try:
        training_data = vn.get_training_data()
        if training_data is None or len(training_data) == 0:
            return jsonify({
                "type": "training_data", 
                "count": 0,
                "message": "No training data found. Run train_vanna.py first"
            })
        
        # Estadísticas
        stats = {
            'total': len(training_data),
            'ddl': len(training_data[training_data['ddl'].notna()]),
            'questions': len(training_data[training_data['question'].notna()]),
            'documentation': len(training_data[training_data['documentation'].notna()])
        }
        
        # Muestras de datos
        sample_questions = training_data[training_data['question'].notna()]['question'].head(5).tolist()
        sample_ddls = training_data[training_data['ddl'].notna()]['ddl'].head(2).tolist()
        
        return jsonify({
            "type": "training_data",
            "stats": stats,
            "sample_questions": sample_questions,
            "sample_ddls": [ddl[:100] + "..." if len(ddl) > 100 else ddl for ddl in sample_ddls]
        })
    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})

@app.route('/')
def root():
    # Verificar estado del entrenamiento
    training_data = vn.get_training_data()
    training_count = len(training_data) if training_data is not None else 0
    
    if training_count == 0:
        status_box = '''
        <div style="color: #d63031; background: #ffe6e6; padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #d63031;">
            <strong>⚠️ Sin Datos de Entrenamiento</strong><br>
            Ejecuta primero: <code style="background: #fff; padding: 2px 5px; border-radius: 3px;">python train_vanna.py</code>
        </div>
        '''
    else:
        ddl_count = len(training_data[training_data['ddl'].notna()])
        q_count = len(training_data[training_data['question'].notna()])
        status_box = f'''
        <div style="color: #00b894; background: #e6fff9; padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #00b894;">
            <strong>✅ Sistema Entrenado</strong><br>
            {training_count} registros cargados (DDL: {ddl_count}, Preguntas: {q_count})
        </div>
        '''
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Vanna + Ollama + MySQL</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", roboto, sans-serif; margin: 40px; background: #f8f9fa; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2d3436; margin-bottom: 10px; }}
            h2 {{ color: #636e72; margin-top: 30px; }}
            .subtitle {{ color: #74b9ff; margin-bottom: 20px; }}
            ul {{ list-style: none; padding: 0; }}
            li {{ margin: 8px 0; }}
            a {{ color: #0984e3; text-decoration: none; padding: 8px 12px; border-radius: 6px; background: #f1f2f6; display: inline-block; }}
            a:hover {{ background: #ddd; }}
            code {{ background: #2d3436; color: #dfe6e9; padding: 2px 6px; border-radius: 4px; }}
            .test-links {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Vanna + Ollama + MySQL</h1>
            <p class="subtitle">Sistema de Generación SQL con IA Local</p>
            
            {status_box}
            
            <h2>🔧 Estado del Sistema</h2>
            <ul>
                <li><a href='/api/v0/test_connection'>🔍 Test Conexiones</a></li>
                <li><a href='/api/v0/tables'>📊 Ver Tablas de BD</a></li>
                <li><a href='/api/v0/training_data'>📚 Datos de Entrenamiento</a></li>
            </ul>
            
            <h2>🧪 Pruebas de Generación SQL</h2>
            <div class="test-links">
                <a href='/api/v0/generate_sql?question=Show all tables'>📋 Mostrar todas las tablas</a>
                <a href='/api/v0/generate_sql?question=List all data'>📄 Listar todos los datos</a>
                <a href='/api/v0/generate_sql?question=Count rows in each table'>🔢 Contar filas en cada tabla</a>
                <a href='/api/v0/generate_sql?question=Show me the first 5 rows'>👀 Primeras 5 filas</a>
            </div>
            
            <h2>📖 Cómo usar</h2>
            <ol>
                <li><strong>Entrenar:</strong> <code>python train_vanna.py</code></li>
                <li><strong>Ejecutar:</strong> <code>python app.py</code></li>
                <li><strong>Generar SQL:</strong> <code>/api/v0/generate_sql?question=tu_pregunta</code></li>
                <li><strong>Ejecutar SQL:</strong> <code>/api/v0/run_sql?id=QUERY_ID</code></li>
            </ol>
            
            <h2>⚡ Información Técnica</h2>
            <ul>
                <li><strong>Base de Datos:</strong> {os.environ.get('MYSQL_DATABASE', 'N/A')}</li>
                <li><strong>Modelo Ollama:</strong> {os.environ.get('OLLAMA_MODEL', 'llama3.2:latest')}</li>
                <li><strong>Almacenamiento:</strong> ./vanna_local_storage/</li>
            </ul>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print(f"\n🌐 Servidor disponible en: http://localhost:5001")
    print(f"📊 Base de datos: {os.environ.get('MYSQL_DATABASE', 'N/A')}")
    print(f"🤖 Modelo Ollama: {os.environ.get('OLLAMA_MODEL', 'llama3.2:latest')}")
    
    training_count = len(vn.get_training_data()) if vn.get_training_data() is not None else 0
    if training_count == 0:
        print(f"\n⚠️  SIN DATOS DE ENTRENAMIENTO")
        print(f"   Ejecuta primero: python train_vanna.py")
    else:
        print(f"\n✅ {training_count} registros de entrenamiento cargados")
    
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)