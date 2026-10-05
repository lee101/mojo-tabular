import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from predict import Predictor

MAX_BODY = 4 * 1024 * 1024
P = Predictor()
STATE = {'status': 'SETUP'}
SCHEMA = {'outputKind': 'json', 'inputs': [
    {'name': 'dataset', 'type': 'string', 'required': True, 'order': 0,
     'choices': ['adult', 'bank-marketing', 'credit-g', 'diabetes', 'phoneme'],
     'description': 'Model to use (OpenML dataset schema)'},
    {'name': 'rows', 'type': 'string', 'required': True, 'order': 1,
     'description': 'JSON object or list of up to 1000 objects with the dataset columns'}]}


class H(BaseHTTPRequestHandler):
    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == '/health-check':
            self.send(200, {'status': STATE['status']})
        elif self.path == '/openapi.json':
            self.send(200, SCHEMA)
        else:
            self.send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != '/predictions':
            return self.send(404, {'error': 'not found'})
        n = int(self.headers.get('content-length') or 0)
        if n > MAX_BODY:
            return self.send(413, {'status': 'failed', 'error': 'body too large'})
        try:
            inp = json.loads(self.rfile.read(n))['input']
            out = P.predict(inp['dataset'], inp['rows'])
            self.send(200, {'status': 'succeeded', 'output': json.loads(out)})
        except Exception as e:
            self.send(200, {'status': 'failed', 'error': f'{type(e).__name__}: {e}'[:300]})

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    P.setup()
    STATE['status'] = 'READY'
    ThreadingHTTPServer(('0.0.0.0', 5000), H).serve_forever()
