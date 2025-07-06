# laboratorio_vulnerabilidades.py
from flask import Flask, request, render_template_string, redirect, url_for
import sqlite3
from werkzeug.security import generate_password_hash

app = Flask(__name__)

# Configuração do banco de dados SQLite
def init_db():
    conn = sqlite3.connect('vulnerable.db')
    c = conn.cursor()
    
    # Criar tabela de usuários
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
    
    # Inserir usuário de teste (senha: "123456")
    hashed_password = generate_password_hash("123456", method='pbkdf2:sha256')
    c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", 
              ("admin", hashed_password))
    
    # Criar tabela de comentários
    c.execute('''CREATE TABLE IF NOT EXISTS comments
                 (id INTEGER PRIMARY KEY, content TEXT)''')
    
    conn.commit()
    conn.close()

init_db()

# Rotas da aplicação vulnerável
@app.route('/')
def index():
    return '''
        <h1>Laboratório de Vulnerabilidades Web</h1>
        <ul>
            <li><a href="/login">Login (SQL Injection)</a></li>
            <li><a href="/comentarios">Comentários (XSS)</a></li>
            <li><a href="/file">Download de Arquivos (Path Traversal)</a></li>
            <li><a href="/testes">Executar Testes Automáticos</a></li>
        </ul>
    '''

# 1. Vulnerabilidade: SQL Injection
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('vulnerable.db')
        c = conn.cursor()
        
        # Consulta vulnerável a SQL Injection
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        
        try:
            c.execute(query)
            user = c.fetchone()
            
            if user:
                return "<h1>Acesso concedido!</h1><p>Consulta executada: " + query + "</p>"
            else:
                return "<h1>Credenciais inválidas!</h1><p>Consulta executada: " + query + "</p>"
        except Exception as e:
            return f"<h1>Erro na consulta:</h1><p>{str(e)}</p><p>Consulta: {query}</p>"
        finally:
            conn.close()
    
    return '''
        <h1>Login (Teste SQL Injection)</h1>
        <p>Tente: admin' -- (como usuário) e qualquer senha</p>
        <form method="POST">
            Username: <input type="text" name="username"><br>
            Password: <input type="password" name="password"><br>
            <input type="submit" value="Login">
        </form>
    '''

# 2. Vulnerabilidade: XSS (Cross-Site Scripting)
@app.route('/comentarios', methods=['GET', 'POST'])
def comentarios():
    conn = sqlite3.connect('vulnerable.db')
    c = conn.cursor()
    
    if request.method == 'POST':
        comentario = request.form['comentario']
        c.execute("INSERT INTO comments (content) VALUES (?)", (comentario,))
        conn.commit()
    
    # Recuperar todos os comentários
    c.execute("SELECT content FROM comments")
    comentarios = [row[0] for row in c.fetchall()]
    conn.close()
    
    return f'''
        <h1>Comentários (Teste XSS)</h1>
        <p>Tente: &lt;script&gt;alert('XSS');&lt;/script&gt;</p>
        <form method="POST">
            Novo comentário: <input type="text" name="comentario"><br>
            <input type="submit" value="Enviar">
        </form>
        <h2>Comentários anteriores:</h2>
        <ul>
            {"".join(f"<li>{c}</li>" for c in comentarios)}
        </ul>
    '''

# 3. Vulnerabilidade: Path Traversal
@app.route('/file')
def download_file():
    filename = request.args.get('filename', 'documento.txt')
    
    # Simulando leitura de arquivo (não implementado por questões de segurança)
    if '..' in filename or '/' in filename:
        return f'''
            <h1>Path Traversal Detectado!</h1>
            <p>Tentativa de acessar: {filename}</p>
            <p>Em um sistema real, isso poderia permitir acesso a arquivos sensíveis.</p>
        '''
    else:
        return f'''
            <h1>Download de Arquivo</h1>
            <p>Arquivo solicitado: {filename}</p>
            <p>Tente: ?filename=../../etc/passwd</p>
        '''

# 4. Testes Automáticos
@app.route('/testes')
def run_tests():
    import os
    from io import StringIO
    import sys
    import unittest
    
    # Capturar saída dos testes
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    # Classe de testes
    class TestVulnerabilities(unittest.TestCase):
        def setUp(self):
            self.app = app.test_client()
            self.app.testing = True
        
        def test_sql_injection(self):
            # Teste de SQL Injection
            response = self.app.post('/login', data={
                'username': "admin' --",
                'password': 'qualquer'
            })
            self.assertIn(b'Acesso concedido', response.data)
            
        def test_xss(self):
            # Teste de XSS
            payload = "<script>alert('XSS');</script>"
            response = self.app.post('/comentarios', data={
                'comentario': payload
            })
            self.assertIn(payload.encode(), response.data)
            
        def test_path_traversal(self):
            # Teste de Path Traversal
            response = self.app.get('/file?filename=../../etc/passwd')
            self.assertIn(b'Path Traversal Detectado', response.data)
    
    # Executar testes
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestVulnerabilities)
    unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(test_suite)
    
    # Restaurar stdout
    sys.stdout = old_stdout
    test_results = mystdout.getvalue()
    
    return f'''
        <h1>Resultados dos Testes Automáticos</h1>
        <pre>{test_results}</pre>
        <a href="/">Voltar</a>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=5000)