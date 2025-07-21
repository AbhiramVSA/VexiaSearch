
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Terminal, Lock, User, Mail } from "lucide-react";
import { useAuth } from '@/hooks/useAuth';
import { toast } from 'sonner';

const Auth = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [loading, setLoading] = useState(false);
  const { signIn, signUp } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (isLogin) {
        const { error } = await signIn(email, password);
        if (error) {
          toast.error(error.message);
        } else {
          toast.success('Logged in successfully');
          navigate('/');
        }
      } else {
        const { error } = await signUp(email, password, fullName);
        if (error) {
          toast.error(error.message);
        } else {
          toast.success('Account created successfully! Please check your email to verify your account.');
        }
      }
    } catch (error) {
      toast.error('An unexpected error occurred');
    } finally {
      setLoading(false);
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
      <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-zinc-900 border border-zinc-700 flex items-center justify-center">
                <Terminal className="w-4 h-4 text-zinc-400" />
              </div>
              <div>
                <h1 className="text-lg font-mono text-zinc-100">RAG.AUTH</h1>
                <p className="text-xs text-zinc-500 font-mono">v2.4.1</p>
              </div>
            </div>
            <Button variant="ghost" className="text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 font-mono text-sm">
              <a href="/">RETURN</a>
            </Button>
          </div>
        </div>
      </nav>

      <div className="relative z-10 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md mx-auto">
          <div className="mb-8 text-center">
            <Badge variant="outline" className="border-zinc-700 text-zinc-400 font-mono text-xs mb-4">
              <Lock className="w-3 h-3 mr-1" />
              SECURE.ACCESS.PORTAL
            </Badge>
            <h2 className="text-3xl font-mono font-bold text-zinc-100 mb-2">
              {isLogin ? 'ACCESS.GRANTED' : 'ACCOUNT.CREATION'}
            </h2>
            <p className="text-zinc-400 font-mono text-sm">
              {isLogin ? 'Enter credentials to access RAG system' : 'Initialize new user profile'}
            </p>
          </div>

          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-zinc-100 font-mono text-sm flex items-center">
                <User className="w-4 h-4 mr-2" />
                {isLogin ? 'LOGIN.INTERFACE' : 'SIGNUP.INTERFACE'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                {!isLogin && (
                  <div>
                    <label className="block text-zinc-300 font-mono text-xs mb-2">FULL.NAME</label>
                    <Input
                      type="text"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="bg-zinc-800 border-zinc-700 text-zinc-100 font-mono"
                      placeholder="Enter full name..."
                      required={!isLogin}
                    />
                  </div>
                )}
                <div>
                  <label className="block text-zinc-300 font-mono text-xs mb-2">EMAIL.ADDRESS</label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 w-4 h-4 text-zinc-500" />
                    <Input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="bg-zinc-800 border-zinc-700 text-zinc-100 font-mono pl-10"
                      placeholder="user@domain.com"
                      required
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-zinc-300 font-mono text-xs mb-2">PASSWORD</label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-3 w-4 h-4 text-zinc-500" />
                    <Input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="bg-zinc-800 border-zinc-700 text-zinc-100 font-mono pl-10"
                      placeholder="••••••••"
                      required
                    />
                  </div>
                </div>
                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-zinc-100 text-zinc-900 hover:bg-zinc-200 font-mono text-sm"
                >
                  {loading ? 'PROCESSING...' : (isLogin ? 'ACCESS.SYSTEM' : 'CREATE.ACCOUNT')}
                </Button>
              </form>
              
              <div className="mt-6 text-center">
                <button
                  onClick={() => setIsLogin(!isLogin)}
                  className="text-zinc-400 hover:text-zinc-100 font-mono text-xs underline"
                >
                  {isLogin ? 'Need an account? REGISTER.HERE' : 'Have an account? LOGIN.HERE'}
                </button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Auth;
