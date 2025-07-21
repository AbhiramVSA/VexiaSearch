
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Terminal, Database, MessageSquare, Upload, LogOut, User } from "lucide-react";
import { useAuth } from '@/hooks/useAuth';
import { toast } from 'sonner';

const Index = () => {
  const { user, signOut } = useAuth();

  const handleSignOut = async () => {
    const { error } = await signOut();
    if (error) {
      toast.error('Error signing out');
    } else {
      toast.success('Signed out successfully');
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 relative overflow-hidden">
      {/* Technical background */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-1/5 w-px h-20 bg-zinc-800/50"></div>
        <div className="absolute top-1/3 right-1/5 w-16 h-px bg-zinc-700/50"></div>
        <div className="absolute bottom-1/3 left-1/4 w-px h-12 bg-zinc-800/50"></div>
        <div className="absolute top-1/2 left-1/3 w-8 h-px bg-zinc-700/30"></div>
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
                <h1 className="text-lg font-mono text-zinc-100">RAG.SYSTEM</h1>
                <p className="text-xs text-zinc-500 font-mono">v2.4.1</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {user ? (
                <>
                  <div className="flex items-center space-x-2 text-zinc-400 font-mono text-sm">
                    <User className="w-4 h-4" />
                    <span>{user.email}</span>
                  </div>
                  <Button 
                    variant="ghost" 
                    onClick={handleSignOut}
                    className="text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 font-mono text-sm"
                  >
                    <LogOut className="w-4 h-4 mr-2" />
                    LOGOUT
                  </Button>
                </>
              ) : (
                <Button variant="ghost" className="text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 font-mono text-sm">
                  <a href="/auth">LOGIN</a>
                </Button>
              )}
            </div>
          </div>
        </div>
      </nav>

      <div className="relative z-10 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-16">
            <div className="flex justify-center space-x-2 mb-6">
              <Badge variant="outline" className="border-zinc-700 text-zinc-400 font-mono text-xs">
                <Database className="w-3 h-3 mr-1" />
                VECTOR.DATABASE
              </Badge>
              <Badge variant="outline" className="border-zinc-700 text-zinc-400 font-mono text-xs">
                <MessageSquare className="w-3 h-3 mr-1" />
                NEURAL.QUERY
              </Badge>
            </div>
            <h1 className="text-5xl md:text-7xl font-mono font-bold text-zinc-100 mb-6 tracking-tight">
              RAG.PORTAL
            </h1>
            <p className="text-xl text-zinc-400 font-mono max-w-2xl mx-auto leading-relaxed">
              Advanced retrieval-augmented generation system for intelligent document processing and contextual query resolution
            </p>
          </div>

          {/* System Status */}
          <div className="grid md:grid-cols-3 gap-6 mb-16">
            <Card className="bg-zinc-900/30 border-zinc-800 hover:bg-zinc-900/40 transition-colors">
              <CardHeader>
                <CardTitle className="text-zinc-100 font-mono text-sm flex items-center">
                  <Terminal className="w-4 h-4 mr-2 text-green-400" />
                  SYSTEM.STATUS
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="font-mono text-xs text-zinc-400 space-y-1">
                  <div className="flex justify-between">
                    <span>CORE.ENGINE</span>
                    <span className="text-green-400">ONLINE</span>
                  </div>
                  <div className="flex justify-between">
                    <span>VECTOR.DB</span>
                    <span className="text-green-400">ACTIVE</span>
                  </div>
                  <div className="flex justify-between">
                    <span>QUERY.PROC</span>
                    <span className="text-green-400">READY</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-zinc-900/30 border-zinc-800 hover:bg-zinc-900/40 transition-colors">
              <CardHeader>
                <CardTitle className="text-zinc-100 font-mono text-sm flex items-center">
                  <Database className="w-4 h-4 mr-2 text-blue-400" />
                  DEPLOYMENT.STATS
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="font-mono text-xs text-zinc-400 space-y-1">
                  <div className="flex justify-between">
                    <span>DOCUMENTS</span>
                    <span className="text-blue-400">{user ? 'READY' : 'LOGIN.REQ'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>EMBEDDINGS</span>
                    <span className="text-blue-400">{user ? 'ACTIVE' : 'LOGIN.REQ'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>INDEX.SIZE</span>
                    <span className="text-blue-400">{user ? 'DYNAMIC' : 'LOGIN.REQ'}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-zinc-900/30 border-zinc-800 hover:bg-zinc-900/40 transition-colors">
              <CardHeader>
                <CardTitle className="text-zinc-100 font-mono text-sm flex items-center">
                  <MessageSquare className="w-4 h-4 mr-2 text-purple-400" />
                  QUERY.METRICS
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="font-mono text-xs text-zinc-400 space-y-1">
                  <div className="flex justify-between">
                    <span>LATENCY</span>
                    <span className="text-purple-400">{user ? '<100MS' : 'LOGIN.REQ'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>ACCURACY</span>
                    <span className="text-purple-400">{user ? '94.2%' : 'LOGIN.REQ'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>CONTEXT.DEPTH</span>
                    <span className="text-purple-400">{user ? 'MULTI' : 'LOGIN.REQ'}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Action Grid */}
          <div className="grid md:grid-cols-2 gap-8">
            <Card className="bg-zinc-900/50 border-zinc-800 group hover:border-zinc-700 transition-colors">
              <CardHeader>
                <CardTitle className="text-zinc-100 font-mono text-lg flex items-center">
                  <Upload className="w-5 h-5 mr-3 text-orange-400" />
                  DOCUMENT.DEPLOYMENT
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-zinc-400 font-mono text-sm mb-6">
                  Initialize vector database with PDF documents for contextual retrieval and processing
                </p>
                <div className="space-y-3">
                  <div className="flex items-center text-zinc-500 font-mono text-xs">
                    <div className="w-2 h-2 bg-orange-400 mr-2"></div>
                    PDF parsing and extraction
                  </div>
                  <div className="flex items-center text-zinc-500 font-mono text-xs">
                    <div className="w-2 h-2 bg-orange-400 mr-2"></div>
                    Vector embedding generation
                  </div>
                  <div className="flex items-center text-zinc-500 font-mono text-xs">
                    <div className="w-2 h-2 bg-orange-400 mr-2"></div>
                    Database indexing and storage
                  </div>
                </div>
                <Button 
                  className="w-full mt-6 bg-orange-500 hover:bg-orange-600 text-black font-mono" 
                  disabled={!user}
                >
                  <a href={user ? "/deploy" : "/auth"} className="flex items-center justify-center w-full">
                    <Upload className="w-4 h-4 mr-2" />
                    {user ? 'DEPLOY.DOCUMENTS' : 'LOGIN.REQUIRED'}
                  </a>
                </Button>
              </CardContent>
            </Card>

            <Card className="bg-zinc-900/50 border-zinc-800 group hover:border-zinc-700 transition-colors">
              <CardHeader>
                <CardTitle className="text-zinc-100 font-mono text-lg flex items-center">
                  <MessageSquare className="w-5 h-5 mr-3 text-green-400" />
                  QUERY.INTERFACE
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-zinc-400 font-mono text-sm mb-6">
                  Natural language processing with contextual document retrieval and intelligent response generation
                </p>
                <div className="space-y-3">
                  <div className="flex items-center text-zinc-500 font-mono text-xs">
                    <div className="w-2 h-2 bg-green-400 mr-2"></div>
                    Semantic similarity search
                  </div>
                  <div className="flex items-center text-zinc-500 font-mono text-xs">
                    <div className="w-2 h-2 bg-green-400 mr-2"></div>
                    Context-aware responses
                  </div>
                  <div className="flex items-center text-zinc-500 font-mono text-xs">
                    <div className="w-2 h-2 bg-green-400 mr-2"></div>
                    Multi-document reasoning
                  </div>
                </div>
                <Button 
                  className="w-full mt-6 bg-green-500 hover:bg-green-600 text-black font-mono"
                  disabled={!user}
                >
                  <a href={user ? "/chat" : "/auth"} className="flex items-center justify-center w-full">
                    <MessageSquare className="w-4 h-4 mr-2" />
                    {user ? 'ACCESS.QUERY.SYSTEM' : 'LOGIN.REQUIRED'}
                  </a>
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Technical Specifications */}
          <div className="mt-16 text-center">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-zinc-500 font-mono text-xs">
              <div>EMBEDDING.MODEL<br/><span className="text-zinc-400">VECTOR.768D</span></div>
              <div>SIMILARITY.ALGO<br/><span className="text-zinc-400">COSINE.DIST</span></div>
              <div>CONTEXT.WINDOW<br/><span className="text-zinc-400">8K.TOKENS</span></div>
              <div>PROCESSING<br/><span className="text-zinc-400">REAL.TIME</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
