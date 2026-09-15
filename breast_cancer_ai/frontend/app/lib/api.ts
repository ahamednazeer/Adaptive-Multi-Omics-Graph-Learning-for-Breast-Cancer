const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Health
  health: () => req<{ status: string }>('/health'),

  // Overview
  getOverview: () => req<any>('/api/overview'),

  // Pipeline
  runPipeline: (steps?: string[]) =>
    req<any>('/api/pipeline/run', { method: 'POST', body: JSON.stringify({ steps }) }),
  stopPipeline: () => req<any>('/api/pipeline/stop', { method: 'POST' }),
  getPipelineStatus: () => req<any>('/api/pipeline/status'),
  getPipelineSteps:  () => req<any>('/api/pipeline/steps'),

  // Jobs
  getJob: (id: string) => req<any>(`/api/jobs/${id}`),
  listJobs: () => req<any>('/api/jobs/'),

  // Patients
  getPatients: (page = 1, search = '') =>
    req<any>(`/api/patients/?page=${page}&search=${encodeURIComponent(search)}`),
  getPatient: (id: string) => req<any>(`/api/patients/${id}`),
  predictPatient: (data: Record<string, any>) =>
    req<any>('/api/patients/predict', { method: 'POST', body: JSON.stringify(data) }),

  // Biomarkers
  getBiomarkers: (direction = 'all', feature_type = 'all', limit = 30) =>
    req<any>(`/api/biomarkers/?direction=${direction}&feature_type=${feature_type}&limit=${limit}`),

  // Evaluation
  getEvaluation: () => req<any>('/api/evaluation/'),
  getModelComparison: () => req<any>('/api/evaluation/comparison'),
  getModelDetail: (name: string) => req<any>(`/api/evaluation/${encodeURIComponent(name)}`),

  // Validation
  getAllValidation: () => req<any>('/api/validation/'),
  getTemporalValidation: () => req<any>('/api/validation/temporal'),
  getExternalValidation: () => req<any>('/api/validation/external'),
  getResistanceAnalysis: () => req<any>('/api/validation/resistance'),

  // SHAP
  getGlobalSHAP: (limit = 30) => req<any>(`/api/shap/global?limit=${limit}`),
  getPatientSHAP: (id: string) => req<any>(`/api/shap/patient/${id}`),
};

export type { };
