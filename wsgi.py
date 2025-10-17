# wsgi.py
from app import create_app

app = create_app()

if __name__ == "__main__":
    # chỉ để chạy local nếu cần
    app.run(host="0.0.0.0", port=5001, debug=False)
