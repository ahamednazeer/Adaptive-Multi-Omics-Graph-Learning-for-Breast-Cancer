'use client';
import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from 'recharts';

interface ROCPoint { fpr: number; tpr: number; }
interface Props {
  data: ROCPoint[];
  auc?: number;
  color?: string;
  label?: string;
}

export default function ROCChart({ data, auc, color = '#3b82f6', label = 'Model' }: Props) {
  const diagonal = [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }];

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null;
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-sm px-3 py-2 text-xs font-mono">
        <p className="text-slate-400">FPR: {payload[0]?.payload?.fpr?.toFixed(3)}</p>
        <p className="text-blue-400">TPR: {payload[0]?.payload?.tpr?.toFixed(3)}</p>
      </div>
    );
  };

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart margin={{ top: 8, right: 16, bottom: 32, left: 32 }}>
          <XAxis
            dataKey="fpr" type="number" domain={[0, 1]} tickCount={6}
            label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -20, fill: '#64748b', fontSize: 11 }}
            tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            axisLine={{ stroke: '#334155' }} tickLine={false}
          />
          <YAxis
            dataKey="tpr" type="number" domain={[0, 1]} tickCount={6}
            label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', offset: 10, fill: '#64748b', fontSize: 11 }}
            tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            axisLine={{ stroke: '#334155' }} tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line data={diagonal} dataKey="tpr" stroke="#475569" strokeDasharray="4 4" dot={false} name="Random" strokeWidth={1} />
          <Line data={data} dataKey="tpr" stroke={color} dot={false} name={`${label}${auc ? ` (AUC=${auc.toFixed(3)})` : ''}`} strokeWidth={2} />
          <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: '#94a3b8', paddingTop: 16 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
