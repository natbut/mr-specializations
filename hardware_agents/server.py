import cgi
import http.server
import os
import socketserver

# Define the directory to serve
directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server")

class SimpleHTTPRequestHandlerWithUpload(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/upload':
            # Get the content length and parse the form
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            # Parse the form data (which includes the file)
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'})

            # Check if a file is present
            if "file" in form:
                uploaded_file = form["file"]
                file_data = uploaded_file.file.read()

                # Save the uploaded file to the 'server' directory
                with open(os.path.join(directory, uploaded_file.filename), "wb") as f:
                    f.write(file_data)

                # Send response
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"File uploaded successfully!")
            else:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"Error: No file uploaded.")

        else:
            # Handle other POST requests as normal
            super().do_POST()

def start_server():
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    os.chdir(directory)

    # Set the port for the server to run on
    port = 8000
    handler = SimpleHTTPRequestHandlerWithUpload

    # Start the HTTP server
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving directory '{directory}' at http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
