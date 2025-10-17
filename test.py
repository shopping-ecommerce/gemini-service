# """
# Setup và Test Semantic Search với credentials.json
# Chạy: python setup_and_test.py
# """

# import os
# import json
# import vertexai
# from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
# import numpy as np

# # ============= CẤU HÌNH =============
# CREDENTIALS_FILE = "credentials.json"  # Đường dẫn đến file credentials
# LOCATION = "asia-southeast1"  # Hoặc: asia-southeast1

# def load_credentials():
#     """Đọc và validate credentials.json"""
#     print("🔑 Đang kiểm tra credentials...")
    
#     if not os.path.exists(CREDENTIALS_FILE):
#         print(f"❌ Không tìm thấy file: {CREDENTIALS_FILE}")
#         print("\n💡 Hướng dẫn tạo credentials.json:")
#         print("   1. Vào Google Cloud Console")
#         print("   2. IAM & Admin > Service Accounts")
#         print("   3. Tạo Service Account với role: Vertex AI User")
#         print("   4. Tạo key (JSON) và lưu thành credentials.json")
#         return None
    
#     try:
#         with open(CREDENTIALS_FILE, 'r') as f:
#             creds = json.load(f)
        
#         project_id = creds.get('project_id')
#         if not project_id:
#             print("❌ File credentials.json không hợp lệ (thiếu project_id)")
#             return None
        
#         print(f"✅ Credentials hợp lệ")
#         print(f"📍 Project ID: {project_id}")
#         print(f"📍 Service Account: {creds.get('client_email', 'N/A')}")
        
#         return project_id
        
#     except json.JSONDecodeError:
#         print("❌ File credentials.json không đúng định dạng JSON")
#         return None
#     except Exception as e:
#         print(f"❌ Lỗi đọc credentials: {e}")
#         return None

# def setup_vertex_ai(project_id):
#     """Thiết lập Vertex AI với credentials"""
#     print("\n🔧 Đang thiết lập Vertex AI...")
    
#     try:
#         # Set environment variable cho credentials
#         os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_FILE
        
#         # Khởi tạo Vertex AI
#         vertexai.init(project=project_id, location=LOCATION)
        
#         # Load model
#         model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
        
#         print("✅ Vertex AI đã sẵn sàng!")
#         return model
        
#     except Exception as e:
#         print(f"❌ Lỗi setup Vertex AI: {e}")
#         print("\n💡 Có thể do:")
#         print("   1. Chưa enable Vertex AI API")
#         print("   2. Service Account chưa có quyền Vertex AI User")
#         print("   3. Chưa enable billing cho project")
#         return None

# def quick_test(model):
#     """Test nhanh embedding"""
#     print("\n" + "="*60)
#     print("⚡ QUICK TEST")
#     print("="*60)
    
#     test_text = "Hello, this is a test!"
#     print(f"Text: {test_text}")
    
#     try:
#         input_data = TextEmbeddingInput(text=test_text, task_type="RETRIEVAL_DOCUMENT")
#         embedding = model.get_embeddings([input_data])[0]
        
#         print(f"✅ Embedding thành công!")
#         print(f"   - Dimension: {len(embedding.values)}")
#         print(f"   - Sample values: {embedding.values[:3]}")
#         return True
#     except Exception as e:
#         print(f"❌ Lỗi: {e}")
#         return False

# def demo_semantic_search(model):
#     """Demo semantic search hoàn chỉnh"""
#     print("\n" + "="*60)
#     print("🎯 DEMO SEMANTIC SEARCH")
#     print("="*60)
    
#     # Dữ liệu mẫu
#     documents = [
#         "Python là ngôn ngữ lập trình mạnh mẽ cho AI và data science",
#         "Machine learning cho phép máy tính học từ dữ liệu",
#         "Deep learning sử dụng mạng neural nhiều lớp",
#         "TensorFlow là framework phổ biến cho ML",
#         "Google Cloud Platform cung cấp nhiều dịch vụ cloud",
#         "Vertex AI là nền tảng ML thống nhất trên GCP",
#         "JavaScript là ngôn ngữ lập trình cho web",
#         "React giúp xây dựng giao diện người dùng",
#         "Docker giúp containerize ứng dụng",
#         "Kubernetes quản lý container ở quy mô lớn"
#     ]
    
#     print(f"\n📚 Đang index {len(documents)} documents...")
    
#     try:
#         # Tạo embeddings cho documents
#         doc_inputs = [
#             TextEmbeddingInput(text=doc, task_type="RETRIEVAL_DOCUMENT") 
#             for doc in documents
#         ]
#         doc_embeddings = model.get_embeddings(doc_inputs)
#         print("✅ Index hoàn tất!")
        
#         # Test với nhiều queries
#         queries = [
#             "Học máy là gì?",
#             "Framework cho AI",
#             "Công cụ phát triển web",
#             "Nền tảng cloud của Google"
#         ]
        
#         for query in queries:
#             print(f"\n{'='*60}")
#             print(f"🔍 Query: '{query}'")
#             print("-" * 60)
            
#             # Tạo embedding cho query
#             query_input = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
#             query_embedding = model.get_embeddings([query_input])[0]
            
#             # Tính cosine similarity
#             query_vec = np.array(query_embedding.values)
#             similarities = []
            
#             for doc_emb in doc_embeddings:
#                 doc_vec = np.array(doc_emb.values)
#                 sim = np.dot(query_vec, doc_vec) / (
#                     np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
#                 )
#                 similarities.append(sim)
            
#             # Hiển thị top 3 kết quả
#             top_indices = np.argsort(similarities)[-3:][::-1]
            
#             for i, idx in enumerate(top_indices, 1):
#                 score = similarities[idx]
#                 emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
#                 print(f"{emoji} Top {i}: [Score: {score:.4f}]")
#                 print(f"   {documents[idx]}")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Lỗi: {e}")
#         return False

# def main():
#     print("🚀 GEMINI SEMANTIC SEARCH - SETUP & TEST")
#     print("="*60)
    
#     # Bước 1: Load credentials
#     project_id = load_credentials()
#     if not project_id:
#         return
    
#     # Bước 2: Setup Vertex AI
#     model = setup_vertex_ai(project_id)
#     if not model:
#         return
    
#     # Bước 3: Quick test
#     if not quick_test(model):
#         return
    
#     # Bước 4: Demo semantic search
#     demo_semantic_search(model)
    
#     print("\n" + "="*60)
#     print("✨ SETUP & TEST HOÀN TẤT!")
#     print("="*60)
#     print("\n💡 Bây giờ bạn có thể:")
#     print("   - Import và sử dụng class GeminiSemanticSearch")
#     print("   - Tích hợp vào ứng dụng của bạn")
#     print("   - Scale lên với nhiều documents hơn")

# if __name__ == "__main__":
#     main()