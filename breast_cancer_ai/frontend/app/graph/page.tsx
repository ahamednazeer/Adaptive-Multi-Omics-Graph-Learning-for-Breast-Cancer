'use client';
import React, { useEffect, useRef, useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import EmptyState from '../components/EmptyState';
import { api } from '../lib/api';
import { Graph } from '@phosphor-icons/react';

export default function GraphViewerPage() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [biomarkers, setBiomarkers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [noData, setNoData] = useState(false);

  useEffect(() => {
    api.getBiomarkers('all', 'all', 40)
      .then(d => {
        if (!d?.pipeline_ran || !d?.biomarkers?.length) { setNoData(true); }
        else { setBiomarkers(d.biomarkers); }
      })
      .catch(() => setNoData(true))
      .finally(() => setLoading(false));
  }, []);

  // Simple D3-like SVG force layout using biomarker data
  useEffect(() => {
    if (!biomarkers.length || !svgRef.current) return;

    const svg = svgRef.current;
    const W = svg.clientWidth || 700;
    const H = 480;

    // Known breast cancer gene edges (simplified graph)
    const knownEdges: [string, string][] = [
      ['ERBB2', 'EGFR'], ['ERBB2', 'AKT.S473'], ['ESR1', 'PGR'],
      ['TP53', 'MDM2'], ['MKI67', 'CCND1'], ['AKT.S473', 'MTOR'],
      ['PIK3CA', 'AKT.S473'], ['CDH1', 'CTNNB1'],
    ];

    const geneNames = biomarkers.slice(0, 30).map(b => b.gene);
    const positions: Record<string, { x: number; y: number; color: string }> = {};
    const cx = W / 2, cy = H / 2;

    geneNames.forEach((g, i) => {
      const angle = (i / geneNames.length) * 2 * Math.PI;
      const r = 150 + Math.random() * 60;
      positions[g] = {
        x: cx + r * Math.cos(angle),
        y: cy + r * Math.sin(angle),
        color: biomarkers[i].direction === 'positive' ? '#3b82f6' : '#ef4444',
      };
    });

    // Build SVG
    const edges = knownEdges
      .filter(([a, b]) => positions[a] && positions[b])
      .map(([a, b]) => `<line x1="${positions[a].x}" y1="${positions[a].y}" x2="${positions[b].x}" y2="${positions[b].y}" stroke="#334155" stroke-width="1.5" opacity="0.6"/>`)
      .join('');

    const nodes = geneNames.map(g => {
      const p = positions[g];
      const r = 6 + (biomarkers.find(b => b.gene === g)?.mean_abs_shap || 0) * 200;
      return `
        <g>
          <circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${p.color}" opacity="0.8" stroke="${p.color}" stroke-width="1"/>
          <text x="${p.x}" y="${p.y + r + 10}" text-anchor="middle" font-size="9" font-family="JetBrains Mono, monospace" fill="#94a3b8">${g.length > 12 ? g.slice(0,12) + '…' : g}</text>
        </g>`;
    }).join('');

    svg.innerHTML = `<g>${edges}${nodes}</g>`;
  }, [biomarkers]);

  return (
    <DashboardLayout pageTitle="Graph Viewer" pageSubtitle="Biological interaction network of selected biomarkers">
      {loading ? (
        <div className="text-center py-16 text-slate-600 font-mono animate-pulse">Loading...</div>
      ) : noData ? (
        <EmptyState title="No Graph Data" message="Run the pipeline through step 5 (STRING Graph Construction) to see the network." icon={<Graph size={48} weight="duotone" />} />
      ) : (
        <div className="space-y-4">
          <div className="flex items-center gap-4 text-xs font-mono text-slate-500">
            <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-blue-500" /><span>pCR Promoter</span></div>
            <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-red-500" /><span>pCR Inhibitor</span></div>
            <div className="flex items-center gap-1.5"><div className="w-4 h-px bg-slate-600" /><span>STRING Interaction</span></div>
            <span className="ml-auto">Node size ∝ SHAP magnitude</span>
          </div>
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm overflow-hidden" style={{ height: 520 }}>
            <svg ref={svgRef} width="100%" height="100%" className="block" />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-4 text-center">
              <p className="text-2xl font-bold font-mono text-slate-100">{biomarkers.length}</p>
              <p className="text-[10px] font-mono text-slate-500 uppercase tracking-wider mt-1">Nodes (Genes)</p>
            </div>
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-4 text-center">
              <p className="text-2xl font-bold font-mono text-blue-400">{biomarkers.filter(b => b.direction === 'positive').length}</p>
              <p className="text-[10px] font-mono text-slate-500 uppercase tracking-wider mt-1">pCR Promoters</p>
            </div>
            <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-4 text-center">
              <p className="text-2xl font-bold font-mono text-red-400">{biomarkers.filter(b => b.direction === 'negative').length}</p>
              <p className="text-[10px] font-mono text-slate-500 uppercase tracking-wider mt-1">pCR Inhibitors</p>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
