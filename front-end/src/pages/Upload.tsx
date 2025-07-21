
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload, FileText, Rocket, Stars, Zap, CheckCircle, Clock, AlertCircle } from "lucide-react";
import { useState } from "react";

const UploadPage = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{[key: string]: number}>({});

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    setFiles(prev => [...prev, ...selectedFiles]);
  };

  const handleUpload = async () => {
    setUploading(true);
    // Simulate upload progress
    for (const file of files) {
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 100));
        setUploadProgress(prev => ({ ...prev, [file.name]: i }));
      }
    }
    setUploading(false);
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-indigo-900 relative overflow-hidden">
      {/* Animated space background */}
      <div className="absolute inset-0">
        <div className="absolute top-10 left-10 w-2 h-2 bg-white rounded-full animate-pulse"></div>
        <div className="absolute top-32 right-20 w-1 h-1 bg-blue-300 rounded-full animate-ping"></div>
        <div className="absolute bottom-40 left-1/4 w-1.5 h-1.5 bg-purple-300 rounded-full animate-pulse"></div>
        <div className="absolute top-1/4 right-1/3 w-1 h-1 bg-white rounded-full animate-ping"></div>
        <div className="absolute bottom-20 right-10 w-2 h-2 bg-indigo-300 rounded-full animate-pulse"></div>
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
                <a href="/demo">Demo</a>
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="relative z-10 py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <Badge variant="secondary" className="mb-4 bg-purple-500/20 text-purple-300 border-purple-500/30">
              <Stars className="w-4 h-4 mr-2" />
              Document Upload Portal
            </Badge>
            <h1 className="text-4xl md:text-6xl font-bold text-white mb-6 animate-fade-in">
              Launch Your
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400"> Documents</span>
              <br />Into the Cosmos
            </h1>
            <p className="text-xl text-purple-200 mb-8 max-w-2xl mx-auto">
              Upload your PDFs to our quantum-powered RAG system and explore the universe of knowledge within your documents.
            </p>
          </div>

          {/* Upload Area */}
          <Card className="bg-black/40 border-purple-500/30 backdrop-blur-sm mb-8 hover:border-purple-400/50 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Upload className="w-6 h-6 mr-2 text-purple-400" />
                Document Upload Station
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-purple-500/50 rounded-lg p-8 text-center hover:border-purple-400 transition-colors duration-300">
                <input
                  type="file"
                  multiple
                  accept=".pdf"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <div className="flex flex-col items-center space-y-4">
                    <div className="w-16 h-16 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-full flex items-center justify-center animate-bounce">
                      <Upload className="w-8 h-8 text-white" />
                    </div>
                    <div>
                      <p className="text-white text-lg font-medium">Drop your PDFs here or click to browse</p>
                      <p className="text-purple-300 text-sm">Support for multiple PDF files</p>
                    </div>
                  </div>
                </label>
              </div>
            </CardContent>
          </Card>

          {/* File List */}
          {files.length > 0 && (
            <Card className="bg-black/40 border-purple-500/30 backdrop-blur-sm mb-8">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <FileText className="w-6 h-6 mr-2 text-cyan-400" />
                  Mission Payload ({files.length} files)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {files.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-purple-900/30 rounded-lg border border-purple-500/20">
                      <div className="flex items-center space-x-3">
                        <FileText className="w-5 h-5 text-cyan-400" />
                        <span className="text-white font-medium">{file.name}</span>
                        <span className="text-purple-300 text-sm">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        {uploadProgress[file.name] !== undefined && (
                          <div className="flex items-center space-x-2">
                            {uploadProgress[file.name] === 100 ? (
                              <CheckCircle className="w-5 h-5 text-green-400" />
                            ) : (
                              <Clock className="w-5 h-5 text-yellow-400 animate-spin" />
                            )}
                            <span className="text-sm text-white">{uploadProgress[file.name]}%</span>
                          </div>
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFile(index)}
                          className="text-red-400 hover:bg-red-500/20"
                        >
                          Remove
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Upload Button */}
          {files.length > 0 && (
            <div className="text-center">
              <Button
                onClick={handleUpload}
                disabled={uploading}
                size="lg"
                className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white px-8 py-4 text-lg font-semibold transform hover:scale-105 transition-all duration-200"
              >
                {uploading ? (
                  <>
                    <Clock className="w-6 h-6 mr-2 animate-spin" />
                    Launching to Cosmos...
                  </>
                ) : (
                  <>
                    <Rocket className="w-6 h-6 mr-2" />
                    Initiate Upload Sequence
                  </>
                )}
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadPage;
