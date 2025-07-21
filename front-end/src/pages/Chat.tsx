
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MessageSquare, Terminal, Send, Brain, Database, User, Bot } from "lucide-react";
import { useState } from "react";
import { useAuth } from '@/hooks/useAuth';
import { ragService } from '@/services/ragService';
import { toast } from 'sonner';

interface Message {
  id: number;
  type: 'system' | 'user' | 'assistant';
  content: string;
  timestamp: string;
}

const Chat = () => {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, type: 'system', content: '$ rag.system initialized successfully', timestamp: '14:32:01' },
    { id: 2, type: 'system', content: '$ vector.database connected [endpoint: ready]', timestamp: '14:32:02' },
    { id: 3, type: 'system', content: '$ query.interface ready for input', timestamp: '14:32:03' }
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || !user) return;
    
    const userMessage: Message = {
      id: Date.now(),
      type: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);
    
    try {
      const response = await ragService.sendMessage(input, user.id);
      
      const botResponse: Message = {
        id: Date.now() + 1,
        type: 'assistant',
        content: response,
        timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
      };
      
      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorResponse: Message = {
        id: Date.now() + 1,
        type: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}. Please check your RAG endpoint configuration.`,
        timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
      };
      
      setMessages(prev => [...prev, errorResponse]);
      toast.error('Failed to get response from RAG system');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 relative overflow-hidden">
      {/* Technical background */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-1/5 w-px h-20 bg-zinc-800/50"></div>
        <div className="absolute top-1/3 right-1/5 w-16 h-px bg-zinc-700/50"></div>
        <div className="absolute bottom-1/3 left-1/4 w-px h-12 bg-zinc-800/50"></div>
      </div>

      {/* Navigation */}
      <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-zinc-900 border border-zinc-700 flex items-center justify-center">
                <Terminal className="w-4 h-4 text-zinc-400" />
              </div>
              <div>
                <h1 className="text-lg font-mono text-zinc-100">RAG.QUERY</h1>
                <p className="text-xs text-zinc-500 font-mono">v2.4.1</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="ghost" className="text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 font-mono text-sm">
                <a href="/">RETURN</a>
              </Button>
              <Button variant="ghost" className="text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 font-mono text-sm">
                <a href="/deploy">DEPLOY.DOCS</a>
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="relative z-10 py-8 px-4 sm:px-6 lg:px-8 h-[calc(100vh-80px)]">
        <div className="max-w-6xl mx-auto h-full flex flex-col">
          {/* Header */}
          <div className="mb-6">
            <div className="flex items-center space-x-2 mb-4">
              <Badge variant="outline" className="border-zinc-700 text-zinc-400 font-mono text-xs">
                <Brain className="w-3 h-3 mr-1" />
                NEURAL.QUERY.INTERFACE
              </Badge>
            </div>
            <h1 className="text-3xl md:text-4xl font-mono font-bold text-zinc-100 mb-2">
              DOCUMENT INTELLIGENCE
            </h1>
            <p className="text-zinc-400 font-mono text-sm">
              Natural language processing with contextual document retrieval
            </p>
          </div>

          {/* Chat interface */}
          <div className="flex-1 grid lg:grid-cols-4 gap-6">
            {/* Chat area */}
            <div className="lg:col-span-3 flex flex-col">
              <Card className="bg-zinc-900/50 border-zinc-800 flex-1 flex flex-col">
                <CardHeader>
                  <CardTitle className="text-zinc-100 font-mono text-sm flex items-center">
                    <MessageSquare className="w-4 h-4 mr-2" />
                    CONVERSATION.LOG
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col">
                  {/* Messages */}
                  <div className="flex-1 space-y-3 mb-4 overflow-y-auto max-h-96">
                    {messages.map((message) => (
                      <div key={message.id} className={`flex items-start space-x-3 ${
                        message.type === 'user' ? 'justify-end' : 'justify-start'
                      }`}>
                        {message.type !== 'user' && (
                          <div className="w-6 h-6 border border-zinc-700 flex items-center justify-center flex-shrink-0">
                            {message.type === 'system' ? (
                              <Terminal className="w-3 h-3 text-zinc-400" />
                            ) : (
                              <Bot className="w-3 h-3 text-zinc-400" />
                            )}
                          </div>
                        )}
                        <div className={`max-w-xs lg:max-w-md ${
                          message.type === 'user' 
                            ? 'bg-zinc-100 text-zinc-900' 
                            : message.type === 'system'
                            ? 'bg-zinc-800/50 text-zinc-400'
                            : 'bg-zinc-800/50 text-zinc-100'
                        } p-3 font-mono text-xs`}>
                          <p>{message.content}</p>
                          <p className="text-xs opacity-60 mt-1">{message.timestamp}</p>
                        </div>
                        {message.type === 'user' && (
                          <div className="w-6 h-6 border border-zinc-700 flex items-center justify-center flex-shrink-0">
                            <User className="w-3 h-3 text-zinc-400" />
                          </div>
                        )}
                      </div>
                    ))}
                    {isProcessing && (
                      <div className="flex items-start space-x-3">
                        <div className="w-6 h-6 border border-zinc-700 flex items-center justify-center">
                          <Bot className="w-3 h-3 text-zinc-400 animate-pulse" />
                        </div>
                        <div className="bg-zinc-800/50 text-zinc-400 p-3 font-mono text-xs">
                          <p>processing query...</p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Input */}
                  <div className="flex space-x-2">
                    <input
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Enter query command..."
                      className="flex-1 bg-zinc-800 border border-zinc-700 text-zinc-100 font-mono text-sm p-3 focus:outline-none focus:border-zinc-600"
                      disabled={isProcessing}
                    />
                    <Button
                      onClick={handleSend}
                      disabled={isProcessing || !input.trim()}
                      className="bg-zinc-100 text-zinc-900 hover:bg-zinc-200 font-mono text-sm px-4"
                    >
                      <Send className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Status panel */}
            <div className="space-y-4">
              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-zinc-100 font-mono text-sm flex items-center">
                    <Database className="w-4 h-4 mr-2" />
                    SYSTEM.STATUS
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between items-center font-mono text-xs">
                    <span className="text-zinc-400">ENDPOINT</span>
                    <span className="text-green-400">READY</span>
                  </div>
                  <div className="flex justify-between items-center font-mono text-xs">
                    <span className="text-zinc-400">USER.AUTH</span>
                    <span className="text-green-400">ACTIVE</span>
                  </div>
                  <div className="flex justify-between items-center font-mono text-xs">
                    <span className="text-zinc-400">VECTOR.DB</span>
                    <span className="text-green-400">ONLINE</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-zinc-100 font-mono text-sm">USER.SESSION</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="font-mono text-xs text-zinc-400 space-y-1">
                    <div className="flex justify-between">
                      <span>USER.ID</span>
                      <span className="text-zinc-500">{user?.id.slice(0, 8)}...</span>
                    </div>
                    <div className="flex justify-between">
                      <span>EMAIL</span>
                      <span className="text-zinc-500">{user?.email?.slice(0, 12)}...</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-zinc-100 font-mono text-sm">QUERY.CONTEXT</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="font-mono text-xs text-zinc-500">
                    <p>Contextual queries enabled</p>
                    <p className="mt-2">Deploy documents for enhanced retrieval</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
