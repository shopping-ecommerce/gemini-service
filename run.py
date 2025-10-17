from dotenv import load_dotenv
load_dotenv()  # <-- nạp .env TRƯỚC

from app import create_app

app = create_app()

if __name__ == "__main__":
    host = app.config.get("HOST", "0.0.0.0")
    port = app.config.get("PORT", 5001)
    print("🚀 Starting Flask Semantic Search API...")
    print("==============================================")
    print(f"📍 API running at: http://{host}:{port}")
    app.run(debug=True, host=host, port=port)
