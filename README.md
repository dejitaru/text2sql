# text2sql
This project convert readable text into sql for mysql. It includes a training module

# How to install
1. Install requirements
```bash
- pip install -r requirements.txt
```
2. Before run, create and activate your virtual environment
```bash
- python -m venv myenv
- source myenv/bin/activate
```
3. Run the training module
```bash
- python train.py
```
4. Run the API to start asking questions
```bash
- python app.py
```
This will start an API on port 5001. Here are the endpoints we have:
[GET] /api/v0/generate_sql
[GET] /api/v0/run_sql
[GET] /api/v0/download_csv
[GET] /api/v0/test_connection
[GET] /api/v0/tables
[GET] /api/v0/training_data

