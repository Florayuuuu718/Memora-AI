import type {
  BestShot,
  EventItem,
  HealthResponse,
  PhotoProject,
  PersonGroup,
  ProjectPhoto,
  SearchResponse,
  SimilarGroup,
  JourneyItem,
} from './types'

export const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const isForm = options.body instanceof FormData
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: isForm
      ? options.headers
      : { 'Content-Type': 'application/json', ...(options.headers || {}) },
  })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      const payload = await response.json()
      message = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail)
    } else {
      message = (await response.text()) || message
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<HealthResponse>('/health'),
  projects: () => request<{ projects: PhotoProject[] }>('/projects'),
  createProject: (name: string) =>
    request<PhotoProject>('/projects', { method: 'POST', body: JSON.stringify({ name }) }),
  uploadFiles: async (projectId: string, files: File[]) => {
    let result: { project: PhotoProject; uploaded: number; rejected: string[] } | null = null
    for (let start = 0; start < files.length; start += 40) {
      const batch = files.slice(start, start + 40)
      const body = new FormData()
      batch.forEach((file) => {
        body.append('files', file)
        body.append('relative_paths', file.webkitRelativePath || file.name)
      })
      const response = await request<{ project: PhotoProject; uploaded: number; rejected: string[] }>(
        `/projects/${projectId}/files`,
        { method: 'POST', body },
      )
      result = result
        ? { project: response.project, uploaded: result.uploaded + response.uploaded, rejected: [...result.rejected, ...response.rejected] }
        : response
    }
    if (!result) throw new Error('No photo files selected')
    return result
  },
  analyze: (projectId: string) =>
    request<{ project: PhotoProject }>(`/projects/${projectId}/analyze`, {
      method: 'POST',
      body: JSON.stringify({ encoder: 'open_clip' }),
    }),
  photos: (projectId: string) =>
    request<{ photos: ProjectPhoto[] }>(`/projects/${projectId}/photos`),
  search: (projectId: string, query: string) =>
    request<SearchResponse>(`/projects/${projectId}/search`, {
      method: 'POST',
      body: JSON.stringify({ query, top_k: 20, strategy: 'query_enhancement' }),
    }),
  events: (projectId: string) =>
    request<{ events: EventItem[] }>(`/projects/${projectId}/events?strategy=strict_event_people`),
  journeys: (projectId: string) => request<{ journeys: JourneyItem[] }>(`/projects/${projectId}/journeys`),
  similar: (projectId: string) =>
    request<{ groups: SimilarGroup[] }>(`/projects/${projectId}/similar-groups`),
  bestShots: (projectId: string) =>
    request<{ photos: BestShot[] }>(`/projects/${projectId}/best-shots`),
  people: (projectId: string) =>
    request<{ groups: PersonGroup[] }>(`/projects/${projectId}/people`),
  clusterPeople: (projectId: string) =>
    request<{ groups: PersonGroup[] }>(`/projects/${projectId}/people/cluster`, {
      method: 'POST',
      body: JSON.stringify({ model_name: 'buffalo_l', eps: 0.35, min_samples: 2 }),
    }),
  saveAnnotation: (projectId: string, kind: 'people' | 'events' | 'journeys', id: number, payload: { name?: string; note?: string; auto_generate_name?: boolean; auto_generate_note?: boolean }) =>
    request<{ annotation: { name: string | null; note: string | null } }>(`/projects/${projectId}/annotations/${kind}/${id}`, { method: 'PUT', body: JSON.stringify(payload) }),
  generateAnnotation: (projectId: string, kind: 'people' | 'events' | 'journeys', id: number, field: 'name' | 'note') =>
    request<{ field: 'name' | 'note'; value: string; source: 'llm' }>(`/projects/${projectId}/annotations/${kind}/${id}/generate`, { method: 'POST', body: JSON.stringify({ field }) }),
}

export function mediaUrl(path?: string): string | undefined {
  if (!path) return undefined
  if (/^https?:\/\//.test(path)) return path
  return `${API_BASE}${path}`
}

export function exportUrl(projectId: string, kind: 'manifest' | 'photos.csv' | 'best-shots.zip'): string {
  return `${API_BASE}/projects/${projectId}/export/${kind}`
}
