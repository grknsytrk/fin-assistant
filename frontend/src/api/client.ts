import type { AskRequest, AskResponse, StatsResponse, FeedbackRequest } from './types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchApi<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE}${endpoint}`;
    const response = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
}

export const apiClient = {
    health: () => fetchApi<{ status: string }>('/health'),

    stats: () => fetchApi<StatsResponse>('/stats'),

    ask: (request: AskRequest) =>
        fetchApi<AskResponse>('/ask', {
            method: 'POST',
            body: JSON.stringify(request),
        }),

    feedback: (request: FeedbackRequest) =>
        fetchApi<{ message: string; path: string; feedback: any }>('/feedback', {
            method: 'POST',
            body: JSON.stringify(request),
        }),

    exportUrl: (type: 'trend' | 'ratio', company?: string) => {
        const params = new URLSearchParams({ type });
        if (company) {
            params.append('company', company);
        }
        return `${API_BASE}/export?${params.toString()}`;
    }
};
