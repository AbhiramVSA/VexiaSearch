
import { useState, useCallback, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Terminal, Upload, FileText, Database, CheckCircle, AlertCircle } from "lucide-react";
import { useAuth } from '@/hooks/useAuth';
import { ragService } from '@/services/ragService';
import { toast } from 'sonner';

const Deploy = () => {
  const { user } = useAuth();
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFiles = Array.from(e.dataTransfer.files).filter(
      file => file.type === 'application/pdf'
    );
    
    if (droppedFiles.length === 0) {
      toast.error('Please upload PDF files only');
      return;
    }

    setFiles(prev => [...prev, ...droppedFiles]);
    toast.success(`Added ${droppedFiles.length} PDF file(s)`);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []).filter(
      file => file.type === 'application/pdf'
    );
    
    if (selectedFiles.length === 0) {
      toast.error('Please select PDF files only');
      return;
    }

    setFiles(prev => [...prev, ...selectedFiles]);
    toast.success(`Added ${selectedFiles.length} PDF file(s)`);
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const deployDocuments = async () => {
    if (!user || files.length === 0) return;

    setUploading(true);
    try {
      const result = await ragService.deployDocuments(files, user.id);
      
      if (result.success) {
        toast.success(`Successfully deployed ${result.documentCount} documents to RAG system`);
        setFiles([]);
        // Redirect to chat after successful deployment
        setTimeout(() => {
          window.location.href = '/chat';
        }, 2000);
      } else {
        toast.error(result.message || 'Failed to deploy documents');
      }
    } catch (error) {
      console.error('Deploy error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to deploy documents');
    } finally {
      setUploading(false);
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
                <h1 className="text-lg font-mono text-zinc-100">RAG.DEPLOY</h1>
                <p className="text-xs text-zinc-500 font-mono">v2.4.1</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="ghost" className="text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 font-mono text-sm">
                <a href="/">RETURN</a>
              </Button>
              <Button variant="ghost" className="text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 font-mono text-sm">
                <a href="/chat">QUERY.INTERFACE</a>
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="relative z-10 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center space-x-2 mb-4">
              <Badge variant="outline" className="border-zinc-700 text-zinc-400 font-mono text-xs">
                <Database className="w-3 h-3 mr-1" />
                VECTOR.DEPLOYMENT.SYSTEM
              </Badge>
            </div>
            <h1 className="text-3xl md:text-4xl font-mono font-bold text-zinc-100 mb-2">
              DOCUMENT DEPLOYMENT
            </h1>
            <p className="text-zinc-400 font-mono text-sm">
              Initialize vector database with PDF documents for intelligent retrieval
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Upload Area */}
            <div className="lg:col-span-2">
              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-zinc-100 font-mono text-sm flex items-center">
                    <Upload className="w-4 h-4 mr-2" />
                    FILE.UPLOAD.INTERFACE
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div
                    className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                      dragActive 
                        ? 'border-zinc-500 bg-zinc-800/50' 
                        : 'border-zinc-700 hover:border-zinc-600'
                    }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                  >
                    <Upload className="w-12 h-12 text-zinc-500 mx-auto mb-4" />
                    <p className="text-zinc-300 font-mono text-sm mb-2">
                      DROP PDF FILES HERE
                    </p>
                    <p className="text-zinc-500 font-mono text-xs mb-4">
                      or click to select files
                    </p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept=".pdf"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                    <Button 
                      variant="outline" 
                      className="border-zinc-700 text-zinc-300 hover:bg-zinc-800 font-mono text-xs"
                      onClick={triggerFileSelect}
                    >
                      SELECT.FILES
                    </Button>
                  </div>

                  {/* File List */}
                  {files.length > 0 && (
                    <div className="mt-6 space-y-2">
                      <h3 className="text-zinc-300 font-mono text-xs mb-3">QUEUED.DOCUMENTS</h3>
                      {files.map((file, index) => (
                        <div key={index} className="flex items-center justify-between bg-zinc-800/50 p-3 rounded border border-zinc-700">
                          <div className="flex items-center space-x-3">
                            <FileText className="w-4 h-4 text-orange-400" />
                            <span className="text-zinc-300 font-mono text-xs">{file.name}</span>
                            <span className="text-zinc-500 font-mono text-xs">
                              {(file.size / 1024 / 1024).toFixed(2)}MB
                            </span>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeFile(index)}
                            className="text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700"
                          >
                            ×
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Deploy Button */}
                  <Button
                    onClick={deployDocuments}
                    disabled={files.length === 0 || uploading}
                    className="w-full mt-6 bg-orange-500 hover:bg-orange-600 text-black font-mono text-sm"
                  >
                    {uploading ? 'DEPLOYING...' : `DEPLOY ${files.length} DOCUMENT${files.length !== 1 ? 'S' : ''}`}
                  </Button>
                </CardContent>
              </Card>
            </div>

            {/* Status Panel */}
            <div className="space-y-4">
              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-zinc-100 font-mono text-sm flex items-center">
                    <Database className="w-4 h-4 mr-2" />
                    SYSTEM.STATUS
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between font-mono text-xs">
                    <span className="text-zinc-400">ENDPOINT.STATUS</span>
                    <div className="flex items-center">
                      <CheckCircle className="w-3 h-3 text-green-400 mr-1" />
                      <span className="text-green-400">READY</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between font-mono text-xs">
                    <span className="text-zinc-400">QUEUE.STATUS</span>
                    <div className="flex items-center">
                      <CheckCircle className="w-3 h-3 text-green-400 mr-1" />
                      <span className="text-green-400">ACTIVE</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between font-mono text-xs">
                    <span className="text-zinc-400">VECTOR.DB</span>
                    <div className="flex items-center">
                      <CheckCircle className="w-3 h-3 text-green-400 mr-1" />
                      <span className="text-green-400">ONLINE</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-zinc-100 font-mono text-sm">DEPLOYMENT.INFO</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="font-mono text-xs text-zinc-500 space-y-2">
                    <p>• PDF content extraction</p>
                    <p>• Vector embedding generation</p>
                    <p>• Database indexing</p>
                    <p>• Context optimization</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-zinc-100 font-mono text-sm flex items-center">
                    <AlertCircle className="w-4 h-4 mr-2 text-yellow-400" />
                    CONFIG.NOTE
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="font-mono text-xs text-zinc-500">
                    <p>Update RAG endpoint URLs in ragService.ts to match your server configuration</p>
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

export default Deploy;
