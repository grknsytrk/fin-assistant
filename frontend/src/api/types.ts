export interface AskRequest {
  question: string;
  retriever?: 'v1' | 'v2' | 'v3' | 'v5' | 'v6';
  mode?: 'single' | 'trend';
  company?: string;
}

export interface AskResponse {
  answer: {
    bullets: string[];
    found: boolean;
    confidence: number;
    verify_status: string;
  };
  parsed: {
    quarter: string | null;
    query_type: string | null;
    company: string | null;
    mentioned_companies?: string[];
  };
  evidence: EvidenceChunk[];
  debug: {
    retriever: string;
    latency_ms: number;
    top_k: number;
  };
  trend?: {
    rows: any[];
  };
  comparison?: {
    mode: string;
    target: string;
    best_company: string | null;
    best_value: number | null;
    best_confidence: number | null;
    rows: any[];
  }
}

export interface EvidenceChunk {
  doc_id: string;
  company: string | null;
  year: string;
  quarter: string;
  page: number;
  section_title: string;
  excerpt: string;
  block_type: string;
  confidence: number | null;
  verify_status: string | null;
  verify_warnings: string[];
}

export interface StatsResponse {
  pdf_count: number;
  page_count: number;
  chunk_count_v1: number;
  chunk_count_v2: number;
  collection_count_v1: number;
  collection_count_v2: number;
  companies: string[];
}

export interface FeedbackRequest {
  timestamp?: string;
  company?: string;
  quarter?: string;
  metric: string;
  extracted_value?: string;
  user_value?: string;
  evidence_ref?: string;
  verdict: 'dogru' | 'yanlis';
}
