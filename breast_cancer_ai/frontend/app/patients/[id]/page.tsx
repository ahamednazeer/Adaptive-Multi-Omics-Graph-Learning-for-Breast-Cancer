'use client';
import React, { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import DashboardLayout from '../../components/DashboardLayout';
import ProbabilityGauge from '../../components/ProbabilityGauge';
import SHAPChart from '../../components/SHAPChart';
import StatusBadge from '../../components/StatusBadge';
import { api } from '../../lib/api';
import { ArrowLeft } from '@phosphor-icons/react';
import Link from 'next/link';

export default function PatientDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [patient, setPatient] = useState<any>(null);
  const [shap, setShap] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getPatient(id).catch(() => null),
      api.getPatientSHAP(id).catch(() => null),
    ]).then(([p, s]) => {
      setPatient(p);
      setShap(s);
      setLoading(false);
    });
  }, [id]);

  if (loading) return (
    <DashboardLayout pageTitle="Patient Detail">
      <div className="text-center py-16 text-slate-600 font-mono animate-pulse">Loading...</div>
    </DashboardLayout>
  );

  if (!patient) return (
    <DashboardLayout pageTitle="Patient Not Found">
      <div className="text-center py-16">
        <p className="font-mono text-slate-500">Patient not found. Run the pipeline first.</p>
        <Link href="/patients" className="mt-4 btn-secondary inline-flex items-center gap-2 text-xs">
          <ArrowLeft size={14} /> Back to Patients
        </Link>
      </div>
    </DashboardLayout>
  );

  return (
    <DashboardLayout pageTitle={`Patient ${patient.patient_id}`} pageSubtitle="Prediction and explanation">
      <Link href="/patients" className="inline-flex items-center gap-2 text-xs font-mono text-slate-500 hover:text-slate-300 mb-6 transition-colors">
        <ArrowLeft size={14} /> All Patients
      </Link>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Prediction card */}
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-sm p-6 flex flex-col items-center gap-4">
          <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500">Prediction</p>
          {patient.probability !== null ? (
            <ProbabilityGauge probability={patient.probability} size="lg" />
          ) : (
            <p className="text-slate-600 font-mono text-sm">No prediction yet</p>
          )}
          <div className="grid grid-cols-2 gap-3 w-full text-center">
            <div className="bg-slate-900/50 rounded-sm p-3 border border-slate-800">
              <p className="text-[10px] font-mono text-slate-500 uppercase">HER2</p>
              <p className="text-sm font-bold font-mono mt-1">{patient.her2 ? 'Positive' : 'Negative'}</p>
            </div>
            <div className="bg-slate-900/50 rounded-sm p-3 border border-slate-800">
              <p className="text-[10px] font-mono text-slate-500 uppercase">HR</p>
              <p className="text-sm font-bold font-mono mt-1">{patient.hr ? 'Positive' : 'Negative'}</p>
            </div>
          </div>
          {patient.treatment_arm && (
            <div className="w-full bg-slate-900/50 rounded-sm p-3 border border-slate-800">
              <p className="text-[10px] font-mono text-slate-500 uppercase mb-1">Treatment</p>
              <p className="text-xs font-mono text-slate-300">{patient.treatment_arm}</p>
            </div>
          )}
        </div>

        {/* SHAP features */}
        <div className="lg:col-span-2 bg-slate-800/40 border border-slate-700/60 rounded-sm p-6">
          <p className="font-chivo font-bold text-xs uppercase tracking-widest text-slate-500 mb-4">SHAP Feature Importance</p>
          {shap?.features?.length > 0 ? (
            <SHAPChart features={shap.features.slice(0, 20).map((f: any) => ({ name: f.name, value: f.shap_value, type: f.type }))} />
          ) : (
            <p className="text-slate-600 font-mono text-sm text-center py-8">No SHAP data available for this patient.</p>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
