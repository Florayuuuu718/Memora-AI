export interface HealthResponse {
  status: string
  photos: number
  encoder: string
  immich_configured: boolean
  llm_configured: boolean
}

export interface PhotoProject {
  id: string
  name: string
  created_at: string
  updated_at: string
  photo_count: number
  analyzed_count: number
  status: 'empty' | 'uploaded' | 'analyzing' | 'ready'
  encoder: 'lightweight' | 'open_clip' | null
  embedding_dimension: number | null
}

export interface ProjectPhoto {
  id: string
  filename: string
  relative_path: string
  captured_at: string | null
  width: number
  height: number
  quality_score: number | null
  url: string
}

export interface SearchResult {
  photo_id: string
  path: string
  score: number
  captured_at: string | null
  source: string
  thumbnail_url?: string
}

export interface SearchResponse {
  total_count: number
  candidate_count: number
  results: SearchResult[]
}

export interface EventItem {
  id: number
  photo_ids: string[]
  start: string | null
  end: string | null
  name?: string | null
  summary?: string | null
  activity_tags?: string[]
  person_ids?: number[]
  note?: string | null
  name_source?: string | null
}

export interface JourneyItem {
  id: number
  event_ids: number[]
  photo_ids: string[]
  start: string | null
  end: string | null
  name?: string | null
  note?: string | null
  person_ids?: number[]
  destination_names?: string[]
}

export interface SimilarGroup {
  id: number
  photo_ids: string[]
  representative_id: string | null
}

export interface PersonGroup {
  id: number
  photo_ids: string[]
  name: string | null
  cover_photo_id?: string | null
  cover_url?: string | null
  note?: string | null
}

export interface BestShot {
  id: string
  filename: string
  score: number
  url: string
}
