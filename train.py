#!/usr/bin/env python3
"""
Script para entrenar Vanna LOCAL con Ollama (sin servicios remotos)
Uso: python train_vanna.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import requests
import json
import pickle
from pathlib import Path

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
                print(f"✅ Cargados {len(data)} registros de entrenamiento existentes")
                return data
            except Exception as e:
                print(f"⚠️  Error cargando training data: {e}")
        
        # Si no existe, crear vacío
        return pd.DataFrame(columns=["question", "sql", "ddl", "documentation"])
    
    def _save_training_data(self):
        """Guarda los datos de entrenamiento en archivo local"""
        try:
            with open(self.training_file, 'wb') as f:
                pickle.dump(self._training_data, f)
        except Exception as e:
            print(f"⚠️  Error guardando training data: {e}")
    
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
    
    def train(self, question=None, sql=None, ddl=None, documentation=None):
        """Añade datos de entrenamiento"""
        new_row = {
            "question": question,
            "sql": sql, 
            "ddl": ddl,
            "documentation": documentation
        }
        
        # Añadir a DataFrame
        self._training_data = pd.concat([
            self._training_data, 
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        # Guardar automáticamente
        self._save_training_data()
    
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
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"❌ Error conectando con Ollama: {e}")
            return f"Error: No se pudo conectar con Ollama - {str(e)}"
    
    def generate_sql(self, question: str):
        """Genera SQL basado en la pregunta y datos de entrenamiento"""
        
        # Construir contexto desde datos de entrenamiento
        context_parts = []
        
        # Añadir DDLs (esquemas de tablas)
        ddls = self._training_data[self._training_data['ddl'].notna()]['ddl'].tolist()
        if ddls:
            context_parts.append("Database Schema:")
            for ddl in ddls[:5]:  # Limitar a 5 tablas para no sobrecargar
                context_parts.append(ddl)
        
        # Añadir ejemplos de questions/sql
        examples = self._training_data[
            (self._training_data['question'].notna()) & 
            (self._training_data['sql'].notna())
        ][['question', 'sql']].head(10)  # Limitar ejemplos
        
        if len(examples) > 0:
            context_parts.append("\nExample Questions and SQL:")
            for _, row in examples.iterrows():
                context_parts.append(f"Q: {row['question']}")
                context_parts.append(f"SQL: {row['sql']}")
        
        # Añadir documentación
        docs = self._training_data[self._training_data['documentation'].notna()]['documentation'].tolist()
        if docs:
            context_parts.append("\nAdditional Information:")
            context_parts.extend(docs[:3])  # Limitar documentación
        
        # Construir prompt
        context = "\n".join(context_parts)
        
        prompt = f"""Given the following database schema and examples, generate a SQL query to answer the question.

{context}

Question: {question}

Generate only the SQL query, no explanations. The query should be valid MySQL syntax."""

        # Obtener respuesta de Ollama
        response = self.submit_prompt(prompt)
        
        # Limpiar respuesta (remover texto extra)
        sql = self._clean_sql_response(response)
        return sql
    
    def _clean_sql_response(self, response: str) -> str:
        """Limpia la respuesta de Ollama para extraer solo el SQL"""
        lines = response.strip().split('\n')
        
        # Buscar líneas que parezcan SQL
        sql_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Buscar líneas que empiecen con palabras clave SQL
            if any(line.upper().startswith(keyword) for keyword in 
                   ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'SHOW', 'DESCRIBE', 'DESC', 'CREATE', 'ALTER', 'DROP']):
                sql_lines.append(line)
            elif line.startswith('```sql'):
                continue
            elif line.startswith('```'):
                continue
            elif sql_lines and not line.startswith(('--', '#', '//')):
                # Si ya tenemos SQL y la línea no es comentario, podría ser continuación
                sql_lines.append(line)
        
        if sql_lines:
            sql = ' '.join(sql_lines)
            # Asegurar que termine con ;
            if not sql.endswith(';'):
                sql += ';'
            return sql
        
        # Si no encontramos SQL claro, devolver la respuesta original
        return response.strip()

def train_basic_sql_patterns(vn):
    """Entrena patrones básicos de SQL"""
    
    basic_patterns = [
        {
            "question": "Show all tables",
            "sql": "SHOW TABLES;",
            "documentation": "Use SHOW TABLES to list all tables in the database"
        },
        {
            "question": "Describe table structure", 
            "sql": "DESCRIBE table_name;",
            "documentation": "Use DESCRIBE or DESC to see table structure"
        },
        {
            "question": "Count all rows",
            "sql": "SELECT COUNT(*) FROM table_name;",
            "documentation": "Use COUNT(*) to count total rows in a table"
        },
        {
            "question": "Show first 10 rows",
            "sql": "SELECT * FROM table_name LIMIT 10;",
            "documentation": "Use LIMIT to restrict number of results"
        },
        {
            "question": "Show unique values",
            "sql": "SELECT DISTINCT column_name FROM table_name;",
            "documentation": "Use DISTINCT to get unique values"
        },
         {
            "question": "Usuarios mayores de edad",
            "sql": "SELECT * FROM users WHERE DATE_ADD(birthday, INTERVAL 18 YEAR) <= CURDATE();",
            "documentation": "Use birthday to get user age"
        }
    ]
    
    print("📚 Entrenando patrones básicos de SQL...")
    for i, pattern in enumerate(basic_patterns, 1):
        vn.train(
            question=pattern["question"],
            sql=pattern["sql"],
            documentation=pattern["documentation"]
        )
        print(f"   [{i}/{len(basic_patterns)}] {pattern['question']}")
    
    print(f"✅ {len(basic_patterns)} patrones básicos añadidos")

def get_table_schema(vn, table_name):
    """Obtiene el esquema completo de una tabla"""
    try:
        # Obtener estructura básica
        describe_df = vn.run_sql(f"DESCRIBE {table_name}")
        
        # Crear DDL desde DESCRIBE
        create_statement = f"CREATE TABLE {table_name} (\n"
        for _, row in describe_df.iterrows():
            field = row['Field']
            type_info = row['Type']
            null_info = "NOT NULL" if row['Null'] == 'NO' else "NULL"
            key_info = f" PRIMARY KEY" if row['Key'] == 'PRI' else ""
            default_info = f" DEFAULT {row['Default']}" if pd.notna(row['Default']) else ""
            create_statement += f"  {field} {type_info} {null_info}{key_info}{default_info},\n"
        create_statement = create_statement.rstrip(',\n') + "\n);"
        
        return create_statement, describe_df
        
    except Exception as e:
        print(f"❌ Error obteniendo esquema de {table_name}: {e}")
        return None, None

def train_table_specific_queries(vn, table_name, describe_df):
    """Genera queries específicas para una tabla"""
    
    columns = describe_df['Field'].tolist()
    
    # Queries básicas para la tabla
    table_queries = [
        {
            "question": f"Show all data from {table_name}",
            "sql": f"SELECT * FROM {table_name};"
        },
        {
            "question": f"Show all {table_name}",
            "sql": f"SELECT * FROM {table_name};"
        },
        {
            "question": f"List all {table_name}",
            "sql": f"SELECT * FROM {table_name};"
        },
        {
            "question": f"Count rows in {table_name}",
            "sql": f"SELECT COUNT(*) as total_rows FROM {table_name};"
        },
        {
            "question": f"Show first 10 {table_name}",
            "sql": f"SELECT * FROM {table_name} LIMIT 10;"
        }
    ]
    
    # Queries para columnas específicas (máximo 3 para no sobrecargar)
    primary_columns = columns[:3]
    for col in primary_columns:
        table_queries.extend([
            {
                "question": f"Show {col} from {table_name}",
                "sql": f"SELECT {col} FROM {table_name};"
            },
            {
                "question": f"Show distinct {col} from {table_name}",
                "sql": f"SELECT DISTINCT {col} FROM {table_name};"
            }
        ])
    
    # Añadir queries al entrenamiento
    for i, query in enumerate(table_queries, 1):
        vn.train(question=query["question"], sql=query["sql"])
        if i % 5 == 0:  # Mostrar progreso cada 5 queries
            print(f"      [{i}/{len(table_queries)}] queries procesadas")
    
    return len(table_queries)

def main():
    """Función principal de entrenamiento"""
    
    print("🚀 Iniciando entrenamiento LOCAL de Vanna + Ollama...")
    print("=" * 60)
    
    # Verificar que pymysql esté instalado
    try:
        import pymysql
        print("✅ PyMySQL disponible")
    except ImportError:
        print("❌ Instalando PyMySQL...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pymysql'])
        import pymysql
        print("✅ PyMySQL instalado")
    
    # Verificar variables de entorno
    required_vars = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_DATABASE']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Variables de entorno faltantes: {missing_vars}")
        return False
    
    print(f"🔧 Configuración:")
    print(f"   Host: {os.environ.get('MYSQL_HOST')}")
    print(f"   Usuario: {os.environ.get('MYSQL_USER')}")
    print(f"   Database: {os.environ.get('MYSQL_DATABASE')}")
    print(f"   Ollama: {os.environ.get('OLLAMA_MODEL', 'llama3.2:latest')}")
    
    # Crear instancia de Vanna local
    vn = LocalVanna(
        model_name=os.environ.get("OLLAMA_MODEL", "llama3.2:latest"),
        storage_path="./vanna_local_storage"
    )
    
    # Conectar a MySQL
    try:
        vn.connect_mysql()
        print("✅ Conectado a MySQL")
        
        # Test conexión
        test_result = vn.run_sql("SELECT 1 as test")
        print(f"✅ Test SQL exitoso: {len(test_result)} filas")
    except Exception as e:
        print(f"❌ Error conectando a MySQL: {e}")
        return False
    
    # Obtener tablas
    try:
        tables_df = vn.run_sql("SHOW TABLES")
        table_column = tables_df.columns[0]
        tables = tables_df[table_column].tolist()
        print(f"🔍 Encontradas {len(tables)} tablas: {tables}")
        
        if not tables:
            print("⚠️  No hay tablas en la base de datos")
            return False
            
    except Exception as e:
        print(f"❌ Error obteniendo tablas: {e}")
        return False
    
    # Limpiar datos de entrenamiento anteriores (opcional)
    clear_previous = input("\n❓ ¿Limpiar datos de entrenamiento anteriores? (y/N): ").lower().strip()
    if clear_previous == 'y':
        vn._training_data = pd.DataFrame(columns=["question", "sql", "ddl", "documentation"])
        vn._save_training_data()
        print("🗑️  Datos anteriores limpiados")
    
    # Entrenar patrones básicos
    print("\n" + "="*40)
    train_basic_sql_patterns(vn)
    
    # Entrenar con cada tabla
    total_queries = 0
    successful_tables = 0
    
    print("\n" + "="*40)
    print("📊 PROCESANDO TABLAS")
    
    for i, table in enumerate(tables, 1):
        print(f"\n[{i}/{len(tables)}] 📋 Tabla: {table}")
        
        # Obtener esquema
        create_statement, describe_df = get_table_schema(vn, table)
        
        if create_statement and describe_df is not None:
            # Añadir DDL
            vn.train(ddl=create_statement)
            print(f"   ✅ DDL añadido")
            
            # Añadir queries específicas
            query_count = train_table_specific_queries(vn, table, describe_df)
            total_queries += query_count
            successful_tables += 1
            print(f"   ✅ {query_count} queries añadidas")
            
            # Añadir documentación
            columns = describe_df['Field'].tolist()
            doc = f"Table '{table}' contains columns: {', '.join(columns)}"
            vn.train(documentation=doc)
            print(f"   ✅ Documentación añadida")
            
        else:
            print(f"   ❌ Error procesando tabla {table}")
    
    # Documentación general
    db_name = os.environ['MYSQL_DATABASE']
    general_doc = f"This MySQL database '{db_name}' contains {len(tables)} tables: {', '.join(tables)}"
    vn.train(documentation=general_doc)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("🎓 ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"✅ Tablas procesadas: {successful_tables}/{len(tables)}")
    print(f"✅ Queries de entrenamiento: {total_queries}")
    
    training_data = vn.get_training_data()
    if training_data is not None:
        print(f"✅ Total registros: {len(training_data)}")
        
        # Mostrar estadísticas
        stats = {
            'DDL': len(training_data[training_data['ddl'].notna()]),
            'Questions': len(training_data[training_data['question'].notna()]),
            'Documentation': len(training_data[training_data['documentation'].notna()])
        }
        
        for category, count in stats.items():
            print(f"   - {category}: {count}")
    
    # Test de generación SQL
    print(f"\n🧪 TESTS DE GENERACIÓN SQL")
    print("=" * 40)
    
    test_questions = [
        "Show all tables",
        f"Show all data from {tables[0]}" if tables else "SELECT 1",
        "Count all rows in each table"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] Pregunta: {question}")
        try:
            sql = vn.generate_sql(question)
            print(f"    SQL: {sql}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Información final
    print(f"\n🎉 ¡ENTRENAMIENTO EXITOSO!")
    print(f"📁 Datos guardados en: {vn.storage_path}")
    print(f"🗃️  Archivo de entrenamiento: {vn.training_file}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n💡 PRÓXIMOS PASOS:")
            print("1. python app.py              # Ejecutar servidor")
            print("2. http://localhost:5001      # Abrir en navegador")
            print("3. Prueba generar SQL!")
            exit(0)
        else:
            exit(1)
    except KeyboardInterrupt:
        print("\n❌ Entrenamiento cancelado")
        exit(1)
    except Exception as e:
        print(f"\n💥 Error fatal: {e}")
        import traceback
        traceback.print_exc()
        exit(1)