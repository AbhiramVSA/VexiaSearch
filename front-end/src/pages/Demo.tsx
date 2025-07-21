
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { MessageSquare, Send, Rocket, Stars, Bot, User, Sparkles } from "lucide-react";
import { useState } from "react";

const DemoPage = () => {
  const [messages, setMessages] = useState([
    { id: 1, type: "bot", content: "Welcome to the Cosmic RAG Demo! I'm your AI assistant ready to help you explore the universe of knowledge. What would you like to know?", timestamp: "10:00 AM" },
    { id: 2, type: "user", content: "Tell me about quantum computing", timestamp: "10:01 AM" },
    { id: 3, type: "bot", content: "Quantum computing is a revolutionary technology that harnesses the principles of quantum mechanics to process information in ways that classical computers cannot. Unlike traditional bits that exist in states of 0 or 1, quantum bits (qubits) can exist in superposition, allowing them to be in multiple states simultaneously.", timestamp: "10:02 AM" }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const sendMessage = () => {
    if (!inputMessage.trim()) return;

    const newMessage = {
      id: messages.length + 1,
      type: "user" as const,
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, newMessage]);
    setInputMessage("");
    setIsTyping(true);

    // Simulate bot response
    setTimeout(() => {
      const botResponse = {
        id: messages.length + 2,
        type: "bot" as const,
        content: "That's an excellent question! Based on the documents in our cosmic knowledge base, I can provide you with detailed insights. This is a demo response showing how our RAG system would analyze your documents and provide contextual answers.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, botResponse]);
      setIsTyping(false);
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-indigo-900 relative overflow-hidden">
      {/* Animated space background */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-16 w-2 h-2 bg-white rounded-full animate-pulse"></div>
        <div className="absolute top-40 right-32 w-1 h-1 bg-cyan-300 rounded-full animate-ping"></div>
        <div className="absolute bottom-32 left-1/3 w-1.5 h-1.5 bg-purple-300 rounded-full animate-pulse"></div>
        <div className="absolute top-1/3 right-1/4 w-1 h-1 bg-white rounded-full animate-ping"></div>
        <div className="absolute bottom-40 right-20 w-2 h-2 bg-indigo-300 rounded-full animate-pulse"></div>
      </div>

      {/* Navigation */}
      <nav className="border-b border-purple-500/30 bg-black/30 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-lg flex items-center justify-center animate-pulse">
                <Rocket className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold text-white">Cosmic RAG Portal</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="ghost" className="text-white hover:bg-purple-500/20">
                <a href="/">Home</a>
              </Button>
              <Button variant="ghost" className="text-white hover:bg-purple-500/20">
                <a href="/upload">Upload</a>
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="relative z-10 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <Badge variant="secondary" className="mb-4 bg-cyan-500/20 text-cyan-300 border-cyan-500/30">
              <Sparkles className="w-4 h-4 mr-2" />
              Interactive Demo
            </Badge>
            <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Experience the
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400"> Cosmic Intelligence</span>
            </h1>
            <p className="text-lg text-purple-200 max-w-2xl mx-auto">
              Chat with our AI powered by advanced RAG technology. See how it understands and responds to your queries.
            </p>
          </div>

          {/* Chat Interface */}
          <Card className="bg-black/40 border-purple-500/30 backdrop-blur-sm h-[600px] flex flex-col">
            <CardHeader className="border-b border-purple-500/30">
              <CardTitle className="text-white flex items-center">
                <MessageSquare className="w-6 h-6 mr-2 text-cyan-400" />
                Cosmic Chat Interface
                <div className="ml-auto">
                  <Badge variant="secondary" className="bg-green-500/20 text-green-300 border-green-500/30">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-2"></div>
                    Online
                  </Badge>
                </div>
              </CardTitle>
            </CardHeader>
            
            <CardContent className="flex-1 p-0 flex flex-col">
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.map((message) => (
                  <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`flex items-start space-x-3 max-w-[80%] ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        message.type === 'user' 
                          ? 'bg-gradient-to-br from-purple-600 to-indigo-600' 
                          : 'bg-gradient-to-br from-cyan-600 to-purple-600 animate-pulse'
                      }`}>
                        {message.type === 'user' ? (
                          <User className="w-4 h-4 text-white" />
                        ) : (
                          <Bot className="w-4 h-4 text-white" />
                        )}
                      </div>
                      <div className={`p-4 rounded-lg ${
                        message.type === 'user'
                          ? 'bg-purple-600/30 border border-purple-500/50'
                          : 'bg-cyan-900/30 border border-cyan-500/50'
                      }`}>
                        <p className="text-white">{message.content}</p>
                        <p className="text-xs text-purple-300 mt-2">{message.timestamp}</p>
                      </div>
                    </div>
                  </div>
                ))}
                
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-600 to-purple-600 flex items-center justify-center animate-pulse">
                        <Bot className="w-4 h-4 text-white" />
                      </div>
                      <div className="p-4 rounded-lg bg-cyan-900/30 border border-cyan-500/50">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Input */}
              <div className="p-6 border-t border-purple-500/30">
                <div className="flex space-x-3">
                  <Input
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Ask me anything about the cosmos of knowledge..."
                    className="flex-1 bg-black/50 border-purple-500/30 text-white placeholder:text-purple-300 focus:border-cyan-400"
                  />
                  <Button
                    onClick={sendMessage}
                    disabled={!inputMessage.trim() || isTyping}
                    className="bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-700 hover:to-purple-700 text-white transform hover:scale-105 transition-all duration-200"
                  >
                    <Send className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default DemoPage;
