'use client';
import React, { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import DashboardLayout from '../components/DashboardLayout';
import EmptyState from '../components/EmptyState';
import StatusBadge from '../components/StatusBadge';
import { api } from '../lib/api';
import { MagnifyingGlass, ArrowRight } from '@phosphor-icons/react';

export default function PatientsPage() {
  const router = useRouter();
  const [data, setData] = useState<any>(null);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);

  const fetch = useCallback(async () => {
    setLoading(true);
    const res = await api.getPatients(page, search).catch(() => null);
    setData(res);
    setLoading(false);
  }, [page, search]);

  useEffect(() => { fetch(); }, [fetch]);

  const patients = data?.patients || [];
  const total = data?.total || 0;
  const totalPages = Math.ceil(total / (data?.page_size || 50));

  return (
    <DashboardLayout pageTitle="Patients" pageSubtitle="ISPY2 patient predictions and profiles">
      {/* Search */}
      <div className="flex items-center gap-3 mb-6">
        <div className="relative flex-1 max-w-sm">
          <MagnifyingGlass className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={14} />
          <input
            className="input-modern pl-8 w-full"
            placeholder="Search patient ID..."
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(1); }}
          />
        </div>
        {total > 0 && <p className="text-xs font-mono text-slate-500">{total} patients</p>}
      </div>

      {loading ? (
        <div className="text-center py-16 text-slate-600 font-mono text-sm animate-pulse">Loading...</div>
      ) : patients.length === 0 ? (
        <EmptyState
          title="No Patients Found"
          message="Run the pipeline to generate predictions, or try a different search."
        />
      ) : (
        <>
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm overflow-hidden mb-4">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-900/50">
                  <tr>
                    {['Patient ID','Dataset','HER2','HR','Treatment','Prediction','Probability'].map(h => (
                      <th key={h} className="px-4 py-3 text-left text-xs font-mono text-slate-500 uppercase tracking-wider">{h}</th>
                    ))}
                    <th className="px-4 py-3" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {patients.map((p: any) => (
                    <tr
                      key={p.id}
                      className="hover:bg-slate-800/50 cursor-pointer transition-colors"
                      onClick={() => router.push(`/patients/${p.patient_id}`)}
                    >
                      <td className="px-4 py-3 font-mono text-sm text-slate-200">{p.patient_id}</td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-400">{p.dataset}</td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-300">{p.her2 !== null ? (p.her2 ? 'Pos' : 'Neg') : '—'}</td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-300">{p.hr !== null ? (p.hr ? 'Pos' : 'Neg') : '—'}</td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-400 max-w-[140px] truncate">{p.treatment_arm || '—'}</td>
                      <td className="px-4 py-3">
                        {p.prediction !== null
                          ? <StatusBadge status={p.prediction === 1 ? 'pcr' : 'no-pcr'} />
                          : <span className="text-xs font-mono text-slate-600">—</span>}
                      </td>
                      <td className="px-4 py-3 font-mono text-sm text-slate-200">
                        {p.probability !== null ? `${(p.probability * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="px-4 py-3 text-slate-600"><ArrowRight size={14} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2">
              <button disabled={page === 1} onClick={() => setPage(p => p - 1)} className="btn-secondary text-xs disabled:opacity-40">Prev</button>
              <span className="text-xs font-mono text-slate-500">{page} / {totalPages}</span>
              <button disabled={page === totalPages} onClick={() => setPage(p => p + 1)} className="btn-secondary text-xs disabled:opacity-40">Next</button>
            </div>
          )}
        </>
      )}
    </DashboardLayout>
  );
}
