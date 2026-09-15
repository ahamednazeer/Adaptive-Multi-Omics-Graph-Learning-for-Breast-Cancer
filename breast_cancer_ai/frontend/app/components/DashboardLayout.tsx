'use client';
import React, { useState, useEffect, useRef, useCallback, ReactNode } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Gauge, ArrowsSplit, Users, Sparkle, Dna, Graph, Robot,
  CheckCircle, ChartBar, ChartLineUp, FileText, List, X,
  Brain, Circle, Clock,
} from '@phosphor-icons/react';

const MIN_WIDTH = 60;
const DEFAULT_WIDTH = 64;
const MAX_WIDTH = 300;

const NAV_ITEMS = [
  { icon: Gauge,       label: 'Overview',       path: '/' },
  { icon: ArrowsSplit, label: 'Pipeline',        path: '/pipeline' },
  { icon: Users,       label: 'Patients',        path: '/patients' },
  { icon: Sparkle,     label: 'New Prediction',  path: '/prediction' },
  { icon: Dna,         label: 'Biomarkers',      path: '/biomarkers' },
  { icon: Graph,       label: 'Graph Viewer',    path: '/graph' },
  { icon: Robot,       label: 'Models',          path: '/models' },
  { icon: CheckCircle, label: 'Validation',      path: '/validation' },
  { icon: ChartBar,    label: 'SHAP Explorer',   path: '/shap' },
  { icon: ChartLineUp, label: 'Evaluation',      path: '/evaluation' },
  { icon: FileText,    label: 'Report',          path: '/report' },
];

interface Props {
  children: ReactNode;
  pageTitle?: string;
  pageSubtitle?: string;
}

export default function DashboardLayout({ children, pageTitle, pageSubtitle }: Props) {
  const pathname = usePathname();
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_WIDTH);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [clock, setClock] = useState('');
  const [mobileOpen, setMobileOpen] = useState(false);
  const isResizing = useRef(false);
  const sidebarRef = useRef<HTMLDivElement>(null);

  const isCollapsed = sidebarWidth < 150;

  // Load saved width
  useEffect(() => {
    const saved = localStorage.getItem('sidebar-width');
    if (saved) setSidebarWidth(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, Number(saved))));
  }, []);

  // Clock
  useEffect(() => {
    const tick = () => setClock(new Date().toLocaleTimeString('en-US', { hour12: false }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // API health check
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch('http://localhost:8000/health', { signal: AbortSignal.timeout(3000) });
        setApiOnline(r.ok);
      } catch { setApiOnline(false); }
    };
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);

  // Resize logic
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    const startX = e.clientX;
    const startW = sidebarWidth;
    const onMove = (ev: MouseEvent) => {
      if (!isResizing.current) return;
      const w = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, startW + ev.clientX - startX));
      setSidebarWidth(w);
      localStorage.setItem('sidebar-width', String(w));
    };
    const onUp = () => { isResizing.current = false; window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [sidebarWidth]);

  const isActive = (path: string) => path === '/' ? pathname === '/' : pathname.startsWith(path);

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className={`flex items-center gap-3 px-4 py-5 border-b border-slate-800 ${isCollapsed ? 'justify-center px-2' : ''}`}>
        <Brain size={22} weight="duotone" className="text-blue-400 shrink-0" />
        {!isCollapsed && (
          <div className="overflow-hidden">
            <p className="text-xs font-chivo font-bold uppercase tracking-wider text-slate-100 whitespace-nowrap">Breast Cancer AI</p>
            <p className="text-[9px] font-mono text-slate-500 uppercase tracking-wider whitespace-nowrap">Research Platform</p>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 overflow-y-auto overflow-x-hidden">
        {NAV_ITEMS.map(({ icon: Icon, label, path }) => {
          const active = isActive(path);
          return (
            <Link
              key={path}
              href={path}
              title={isCollapsed ? label : undefined}
              className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors relative group
                ${isCollapsed ? 'justify-center px-3' : ''}
                ${active
                  ? 'text-blue-400 bg-blue-950/50 border-l-2 border-blue-400'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800 border-l-2 border-transparent'
                }`}
            >
              <Icon size={18} weight={active ? 'fill' : 'regular'} className="shrink-0" />
              {!isCollapsed && <span className="font-mono text-xs uppercase tracking-wide whitespace-nowrap">{label}</span>}
              {isCollapsed && (
                <span className="absolute left-full ml-2 bg-slate-800 text-slate-200 text-xs font-mono px-2 py-1 rounded-sm whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none z-50 border border-slate-700">
                  {label}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className={`border-t border-slate-800 p-3 ${isCollapsed ? 'text-center' : ''}`}>
        <p className="text-[10px] font-mono text-slate-600 uppercase tracking-wider">v1.0.0</p>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Scanlines */}
      <div className="scanlines" />

      {/* Sidebar — desktop */}
      <div
        ref={sidebarRef}
        className="hidden md:flex flex-col bg-slate-900 border-r border-slate-800 relative shrink-0 transition-none"
        style={{ width: sidebarWidth }}
      >
        <SidebarContent />
        {/* Resize handle */}
        <div
          className="absolute top-0 right-0 w-1 h-full cursor-ew-resize hover:bg-blue-500/50 transition-colors z-10"
          onMouseDown={onMouseDown}
        />
      </div>

      {/* Mobile sidebar */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setMobileOpen(false)} />
          <div className="absolute left-0 top-0 bottom-0 w-64 bg-slate-900 border-r border-slate-800 flex flex-col">
            <SidebarContent />
          </div>
        </div>
      )}

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="sticky top-0 z-40 backdrop-blur-md bg-slate-950/80 border-b border-slate-800 flex items-center justify-between px-4 h-14 shrink-0">
          <div className="flex items-center gap-4">
            <button className="md:hidden text-slate-400 hover:text-slate-200 p-1" onClick={() => setMobileOpen(v => !v)}>
              <List size={20} />
            </button>
            <div>
              <p className="text-xs font-chivo font-bold uppercase tracking-widest text-slate-100">
                {pageTitle || 'Breast Cancer AI Platform'}
              </p>
              {pageSubtitle && <p className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">{pageSubtitle}</p>}
            </div>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            {/* API status */}
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${apiOnline === true ? 'bg-green-400' : apiOnline === false ? 'bg-red-400 animate-pulse' : 'bg-amber-400 animate-pulse'}`} />
              <span className={`uppercase tracking-wider text-[10px] ${apiOnline === true ? 'text-green-400' : apiOnline === false ? 'text-red-400' : 'text-amber-400'}`}>
                {apiOnline === true ? 'API Online' : apiOnline === false ? 'API Offline' : 'Checking...'}
              </span>
            </div>
            {/* Clock */}
            <div className="hidden sm:flex items-center gap-1 text-slate-500">
              <Clock size={12} />
              <span className="text-[10px] uppercase tracking-wider">{clock}</span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
