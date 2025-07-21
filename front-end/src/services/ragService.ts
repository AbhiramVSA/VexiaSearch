
// Configuration for external RAG endpoints
const RAG_CONFIG = {
  // Update these URLs to match your actual endpoint IPs
  CHAT_ENDPOINT: 'http://127.0.0.1:8000/chat',
  DEPLOY_ENDPOINT: 'http://127.0.0.1:8000/deploy'
};

export interface ChatMessage {
  message: string;
  context?: string;
}

export interface ChatResponse {
  response: string;
  context?: string;
}

export interface DeployResponse {
  success: boolean;
  message: string;
  documentCount?: number;
}

export const ragService = {
  // Send message to RAG chat endpoint
  async sendMessage(message: string, userId: string): Promise<string> {
    try {
      const response = await fetch(RAG_CONFIG.CHAT_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          user_id: userId
        })
      });

      if (!response.ok) {
        throw new Error(`Chat endpoint error: ${response.status}`);
      }

      const data = await response.json();
      return data.response || data.message || 'No response received';
    } catch (error) {
      console.error('RAG Chat Error:', error);
      throw new Error('Failed to get response from RAG system');
    }
  },

  // Deploy PDF files to RAG system
  async deployDocuments(files: File[], userId: string): Promise<DeployResponse> {
    try {
      const formData = new FormData();
      
      files.forEach(file => {
        formData.append(`files`, file);
      });
      
      formData.append('user_id', userId);
      formData.append('file_count', files.length.toString());

      const response = await fetch(RAG_CONFIG.DEPLOY_ENDPOINT, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Deploy endpoint error: ${response.status}`);
      }

      const data = await response.json();
      return {
        success: data.success || true,
        message: data.message || 'Documents deployed successfully',
        documentCount: data.document_count || files.length
      };
    } catch (error) {
      console.error('RAG Deploy Error:', error);
      throw new Error('Failed to deploy documents to RAG system');
    }
  }
};
