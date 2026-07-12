/**
 * Umbuzo - API Client Module
 * Handles all HTTP communication with the backend using Axios
 */

const API_BASE = '';

const UmbuzoAPI = {
    /**
     * Get auth token from storage
     */
    getToken() {
        return localStorage.getItem('umbuzo_token');
    },

    /**
     * Get auth headers
     */
    getHeaders() {
        const token = this.getToken();
        return {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        };
    },

    /**
     * Handle API response
     */
    async handleResponse(response) {
        if (response.status >= 200 && response.status < 300) {
            return response.data;
        }
        const error = response.data?.detail || 'An error occurred';
        throw new Error(error);
    },

    /**
     * Generic request method
     */
    async request(method, url, data = null, options = {}) {
        try {
            const config = {
                method,
                url: `${API_BASE}${url}`,
                headers: this.getHeaders(),
                ...options
            };
            if (data && method !== 'get') {
                config.data = data;
            }
            const response = await axios(config);
            return this.handleResponse(response);
        } catch (error) {
            if (error.response?.status === 401) {
                // Token expired - clear and redirect
                localStorage.removeItem('umbuzo_token');
                localStorage.removeItem('umbuzo_user');
                if (window.location.pathname !== '/login' && window.location.pathname !== '/signup') {
                    window.location.href = '/login';
                }
            }
            const message = error.response?.data?.detail || error.message || 'Request failed';
            throw new Error(message);
        }
    },

    // =========================================================================
    // Authentication
    // =========================================================================

    async signup(username, email, password, fullName = null) {
        const data = await this.request('post', '/api/auth/signup', {
            username,
            email,
            password,
            full_name: fullName
        });
        localStorage.setItem('umbuzo_token', data.access_token);
        localStorage.setItem('umbuzo_user', JSON.stringify(data.user));
        return data;
    },

    async login(username, password) {
        const data = await this.request('post', '/api/auth/login', {
            username,
            password
        });
        localStorage.setItem('umbuzo_token', data.access_token);
        localStorage.setItem('umbuzo_user', JSON.stringify(data.user));
        return data;
    },

    async logout() {
        try {
            await this.request('post', '/api/auth/logout');
        } finally {
            localStorage.removeItem('umbuzo_token');
            localStorage.removeItem('umbuzo_user');
        }
    },

    async getMe() {
        return await this.request('get', '/api/auth/me');
    },

    isAuthenticated() {
        return !!this.getToken();
    },

    getUser() {
        const user = localStorage.getItem('umbuzo_user');
        return user ? JSON.parse(user) : null;
    },

    // =========================================================================
    // Chats
    // =========================================================================

    async createChat(title = null, model = null) {
        return await this.request('post', '/api/chats', { title, model });
    },

    async listChats(skip = 0, limit = 50) {
        return await this.request('get', `/api/chats?skip=${skip}&limit=${limit}`);
    },

    async getChat(chatId) {
        return await this.request('get', `/api/chats/${chatId}`);
    },

    async updateChat(chatId, title = null, model = null) {
        const data = {};
        if (title) data.title = title;
        if (model) data.model = model;
        return await this.request('put', `/api/chats/${chatId}`, data);
    },

    async deleteChat(chatId) {
        return await this.request('delete', `/api/chats/${chatId}`);
    },

    // =========================================================================
    // Messages
    // =========================================================================

    async getMessages(chatId) {
        return await this.request('get', `/api/chats/${chatId}/messages`);
    },

    async sendMessage(chatId, content, attachments = null) {
        return await this.request('post', `/api/chats/${chatId}/messages`, {
            content,
            role: 'user',
            attachments
        });
    },

    // =========================================================================
    // GPTs / Tools
    // =========================================================================

    async listGPTs(category = null) {
        const url = category ? `/api/gpts?category=${category}` : '/api/gpts';
        return await this.request('get', url);
    },

    // =========================================================================
    // Search
    // =========================================================================

    async searchKnowledgeBase(query, topK = 5) {
        return await this.request('get', `/api/search?query=${encodeURIComponent(query)}&top_k=${topK}`);
    },

    // =========================================================================
    // File Upload
    // =========================================================================

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await axios.post(`${API_BASE}/api/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                    ...(this.getToken() ? { 'Authorization': `Bearer ${this.getToken()}` } : {})
                }
            });
            return this.handleResponse(response);
        } catch (error) {
            const message = error.response?.data?.detail || error.message || 'Upload failed';
            throw new Error(message);
        }
    }
};

// Make API globally available
window.UmbuzoAPI = UmbuzoAPI;

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UmbuzoAPI;
}