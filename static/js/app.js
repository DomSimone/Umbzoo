/**
 * Umbuzo - Main Application Module
 * Open-source mode: no sign-up required, works immediately for guests.
 * Handles UI interactions, chat functionality, and page rendering.
 */

const UmbuzoApp = {
    currentChatId: null,
    currentModel: 'gpt-3.5-turbo',
    isLoading: false,
    uploadedFiles: [],

    /**
     * Initialize the application
     */
    async init() {
        // Always start in guest mode - no auth check
        this.renderAuthUI();
        this.bindEvents();

        // Check if we're on a specific page
        const path = window.location.pathname;
        if (path === '/' || path === '/index.html') {
            await this.loadMainPage();
        } else if (path === '/gpts') {
            await this.loadGptsPage();
        } else if (path === '/history') {
            await this.loadHistoryPage();
        }
    },

    /**
     * Render authentication UI (top nav + sidebar)
     */
    renderAuthUI() {
        const isAuth = UmbuzoAPI.isAuthenticated();
        const user = UmbuzoAPI.getUser();

        // Update top nav auth buttons
        const authContainer = document.querySelector('.top-nav-right');
        if (authContainer) {
            if (isAuth && user) {
                authContainer.innerHTML = `
                    <span class="nav-auth-btn" style="color: var(--text-secondary); font-size: 0.8rem;">
                        ${user.username}
                    </span>
                    <button class="nav-auth-btn btn-signup" onclick="UmbuzoApp.handleLogout()">
                        Sign Out
                    </button>
                `;
            } else {
                authContainer.innerHTML = `
                    <a href="/login" class="nav-auth-btn btn-signin">Sign In</a>
                    <a href="/signup" class="nav-auth-btn btn-signup">Sign Up</a>
                `;
            }
        }

        // Update sidebar user info
        const userInfo = document.querySelector('.user-info');
        if (userInfo) {
            if (isAuth && user) {
                const initial = user.username.charAt(0).toUpperCase();
                userInfo.innerHTML = `
                    <div class="user-avatar">${initial}</div>
                    <span class="user-name">${user.username}</span>
                `;
            } else {
                userInfo.innerHTML = `
                    <div class="user-avatar">👤</div>
                    <span class="user-name">Guest</span>
                `;
            }
        }

        // Update active states
        const currentPath = window.location.pathname;
        document.querySelectorAll('.nav-item').forEach(item => {
            const href = item.getAttribute('href');
            item.classList.toggle('active', href === currentPath);
        });
        document.querySelectorAll('.nav-page-btn').forEach(btn => {
            const href = btn.getAttribute('data-href') || btn.getAttribute('href');
            btn.classList.toggle('active', href === currentPath);
        });
    },

    /**
     * Bind global event listeners
     */
    bindEvents() {
        // Menu toggle for mobile
        const menuToggle = document.querySelector('.menu-toggle');
        if (menuToggle) {
            menuToggle.addEventListener('click', () => {
                document.querySelector('.sidebar').classList.toggle('open');
            });
        }

        // Close sidebar on click outside (mobile)
        document.addEventListener('click', (e) => {
            const sidebar = document.querySelector('.sidebar');
            const toggle = document.querySelector('.menu-toggle');
            if (window.innerWidth <= 768 &&
                sidebar && sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) &&
                toggle && !toggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });

        // Handle Enter key in prompt input
        const promptInput = document.querySelector('.prompt-input');
        if (promptInput) {
            promptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.handleSendMessage();
                }
            });

            // Auto-resize textarea
            promptInput.addEventListener('input', () => {
                promptInput.style.height = 'auto';
                promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
                this.updateSendButton();
            });
        }

        // Attachment button
        const attachBtn = document.querySelector('[data-tool="attach"]');
        if (attachBtn) {
            attachBtn.addEventListener('click', () => this.handleFileUpload());
        }

        // Voice mode button
        const voiceBtn = document.querySelector('[data-tool="voice"]');
        if (voiceBtn) {
            voiceBtn.addEventListener('click', () => this.handleVoiceMode());
        }

        // New chat button
        const newChatBtn = document.querySelector('.new-chat-btn');
        if (newChatBtn) {
            newChatBtn.addEventListener('click', () => this.createNewChat());
        }
    },

    /**
     * Load the main chat page
     */
    async loadMainPage() {
        const chatId = this.getChatIdFromUrl();
        if (chatId) {
            this.currentChatId = chatId;
            await this.loadChat(chatId);
        } else {
            this.showWelcomeScreen();
        }

        // Load chats list in sidebar if authenticated
        if (UmbuzoAPI.isAuthenticated()) {
            await this.loadChatList();
        }

        // Update title
        const titleEl = document.querySelector('.top-nav-title');
        if (titleEl && this.currentChatId) {
            try {
                const chat = await UmbuzoAPI.getChat(this.currentChatId);
                titleEl.textContent = chat.title;
            } catch (e) {
                titleEl.textContent = 'Chat';
            }
        } else if (titleEl) {
            titleEl.textContent = 'New Chat';
        }
    },

    /**
     * Load the GPTs directory page
     */
    async loadGptsPage() {
        try {
            const gpts = await UmbuzoAPI.listGPTs();
            this.renderGpts(gpts);
        } catch (e) {
            console.error('Failed to load GPTs:', e);
        }
    },

    /**
     * Load the chat history page
     */
    async loadHistoryPage() {
        if (!UmbuzoAPI.isAuthenticated()) {
            // Show empty state for guests
            this.renderEmptyHistory();
            return;
        }

        try {
            const chats = await UmbuzoAPI.listChats();
            this.renderHistory(chats);
        } catch (e) {
            console.error('Failed to load history:', e);
        }
    },

    /**
     * Render empty history for guest users
     */
    renderEmptyHistory() {
        const list = document.querySelector('.history-list');
        if (!list) return;

        list.innerHTML = `
            <div style="text-align: center; color: var(--text-tertiary); padding: 60px 20px;">
                <p style="font-size: 1.2rem; margin-bottom: 8px;">Sign in to see your history</p>
                <p>Guest chats are not saved. Create an account to persist your conversations.</p>
                <a href="/signup" class="btn-primary" style="display: inline-block; width: auto; margin-top: 20px; padding: 10px 24px; text-decoration: none;">
                    Create Free Account
                </a>
            </div>
        `;
    },

    /**
     * Get chat ID from URL query parameter
     */
    getChatIdFromUrl() {
        const params = new URLSearchParams(window.location.search);
        return params.get('chat_id') ? parseInt(params.get('chat_id')) : null;
    },

    /**
     * Show welcome screen with suggestions
     */
    showWelcomeScreen() {
        const container = document.querySelector('.messages-container');
        if (!container) return;

        const suggestions = [
            { icon: '🌍', text: 'Tell me about African history and culture' },
            { icon: '📊', text: 'Analyze economic trends in East Africa' },
            { icon: '🏛️', text: 'Explain governance structures in Africa' },
            { icon: '🔬', text: 'Research demographic changes in Nigeria' }
        ];

        container.innerHTML = `
            <div class="messages-wrapper">
                <div class="welcome-screen">
                    <img src="/static/images/Mbuzo.png" alt="Umbuzo" class="welcome-logo"
                         onerror="this.style.display='none'">
                    <h1 class="welcome-title">What do you want to know?</h1>
                    <p class="welcome-subtitle">Ask me anything about Africa - history, culture, economics, and more</p>
                    <div class="welcome-suggestions">
                        ${suggestions.map(s => `
                            <div class="suggestion-card" onclick="UmbuzoApp.handleSuggestionClick('${s.text.replace(/'/g, "\\'")}')">
                                <div class="suggestion-icon">${s.icon}</div>
                                <div class="suggestion-text">${s.text}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Handle suggestion click
     */
    handleSuggestionClick(text) {
        const input = document.querySelector('.prompt-input');
        if (input) {
            input.value = text;
            input.dispatchEvent(new Event('input'));
            this.handleSendMessage();
        }
    },

    /**
     * Create a new chat - open to guests too
     */
    async createNewChat() {
        try {
            const chat = await UmbuzoAPI.createChat(null, this.currentModel);
            window.history.pushState({}, '', `/?chat_id=${chat.id}`);
            this.currentChatId = chat.id;
            this.showWelcomeScreen();

            // Update title
            const titleEl = document.querySelector('.top-nav-title');
            if (titleEl) titleEl.textContent = 'New Chat';

            // Update sidebar chat list if authenticated
            if (UmbuzoAPI.isAuthenticated()) {
                await this.loadChatList();
            }
        } catch (e) {
            this.showToast('Failed to create chat: ' + e.message, 'error');
        }
    },

    /**
     * Load a specific chat
     */
    async loadChat(chatId) {
        const container = document.querySelector('.messages-container');
        if (!container) return;

        try {
            this.showLoading(true);
            const messages = await UmbuzoAPI.getMessages(chatId);

            if (!messages || messages.length === 0) {
                this.showWelcomeScreen();
                return;
            }

            container.innerHTML = `
                <div class="messages-wrapper">
                    ${messages.map(m => this.renderMessage(m)).join('')}
                </div>
            `;

            this.scrollToBottom();

            // Update title
            const chat = await UmbuzoAPI.getChat(chatId);
            const titleEl = document.querySelector('.top-nav-title');
            if (titleEl) titleEl.textContent = chat.title;

        } catch (e) {
            console.error('Failed to load chat:', e);
            this.showWelcomeScreen();
        } finally {
            this.showLoading(false);
        }
    },

    /**
     * Render a single message
     */
    renderMessage(message) {
        const time = new Date(message.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const avatarIcon = message.role === 'user' ? '👤' : '🤖';
        const roleName = message.role === 'user' ? 'You' : 'Umbuzo';
        const formattedContent = this.formatMessageContent(message.content);

        return `
            <div class="message">
                <div class="message-avatar ${message.role}">${avatarIcon}</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-role">${roleName}</span>
                        <span class="message-time">${time}</span>
                    </div>
                    <div class="message-body">${formattedContent}</div>
                </div>
            </div>
        `;
    },

    /**
     * Format message content with basic markdown support
     */
    formatMessageContent(content) {
        if (!content) return '';

        let html = content
            .replace(/&/g, '&')
            .replace(/</g, '<')
            .replace(/>/g, '>');

        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');

        return `<p>${html}</p>`;
    },

    /**
     * Handle send message action - open to all
     */
    async handleSendMessage() {
        const input = document.querySelector('.prompt-input');
        if (!input || this.isLoading) return;

        const content = input.value.trim();
        if (!content) return;

        // Clear input
        input.value = '';
        input.style.height = 'auto';
        this.updateSendButton();

        // If no active chat, create one (works for guests too)
        if (!this.currentChatId) {
            try {
                const chat = await UmbuzoAPI.createChat(null, this.currentModel);
                this.currentChatId = chat.id;
                window.history.pushState({}, '', `/?chat_id=${chat.id}`);

                const titleEl = document.querySelector('.top-nav-title');
                if (titleEl) titleEl.textContent = chat.title;

                // Load sidebar if authenticated
                if (UmbuzoAPI.isAuthenticated()) {
                    await this.loadChatList();
                }
            } catch (e) {
                this.showToast('Failed to create chat: ' + e.message, 'error');
                return;
            }
        }

        // Add user message to UI immediately
        const messagesWrapper = document.querySelector('.messages-wrapper') ||
            document.querySelector('.welcome-screen')?.closest('.messages-wrapper');

        if (messagesWrapper) {
            const welcome = messagesWrapper.querySelector('.welcome-screen');
            if (welcome) {
                messagesWrapper.innerHTML = '';
            }

            const userMessage = {
                role: 'user',
                content,
                created_at: new Date().toISOString()
            };
            messagesWrapper.innerHTML += this.renderMessage(userMessage);
        }

        // Show typing indicator
        this.showTypingIndicator();
        this.scrollToBottom();

        // Send message to API
        this.isLoading = true;
        try {
            const response = await UmbuzoAPI.sendMessage(this.currentChatId, content);

            this.removeTypingIndicator();

            if (messagesWrapper) {
                const aiMessage = {
                    role: 'assistant',
                    content: response.content,
                    created_at: response.created_at || new Date().toISOString()
                };
                messagesWrapper.innerHTML += this.renderMessage(aiMessage);
            }

            // Update sidebar if authenticated
            if (UmbuzoAPI.isAuthenticated()) {
                await this.loadChatList();
            }

        } catch (e) {
            this.removeTypingIndicator();
            this.showToast('Failed to send message: ' + e.message, 'error');
        } finally {
            this.isLoading = false;
            this.scrollToBottom();
        }
    },

    /**
     * Show typing indicator
     */
    showTypingIndicator() {
        const container = document.querySelector('.messages-container');
        if (!container) return;

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator-wrapper message';
        indicator.id = 'typing-indicator';
        indicator.innerHTML = `
            <div class="message-avatar assistant">🤖</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-role">Umbuzo</span>
                </div>
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;

        const wrapper = container.querySelector('.messages-wrapper');
        if (wrapper) {
            wrapper.appendChild(indicator);
        }
    },

    /**
     * Remove typing indicator
     */
    removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    },

    /**
     * Update send button state
     */
    updateSendButton() {
        const input = document.querySelector('.prompt-input');
        const sendBtn = document.querySelector('.send-btn');
        if (input && sendBtn) {
            const hasText = input.value.trim().length > 0;
            sendBtn.classList.toggle('active', hasText);
            sendBtn.disabled = !hasText;
        }
    },

    /**
     * Scroll messages container to bottom
     */
    scrollToBottom() {
        const container = document.querySelector('.messages-container');
        if (container) {
            setTimeout(() => {
                container.scrollTop = container.scrollHeight;
            }, 100);
        }
    },

    /**
     * Load chat list into sidebar
     */
    async loadChatList() {
        const chatListEl = document.querySelector('.chat-list');
        if (!chatListEl) return;

        try {
            const chats = await UmbuzoAPI.listChats(0, 20);
            if (chats && chats.length > 0) {
                chatListEl.innerHTML = chats.map(chat => `
                    <div class="chat-list-item ${chat.id === this.currentChatId ? 'active' : ''}"
                         onclick="UmbuzoApp.openChat(${chat.id})"
                         data-chat-id="${chat.id}">
                        <span class="chat-icon">💬</span>
                        <span class="chat-title">${chat.title}</span>
                        <button class="chat-delete" onclick="event.stopPropagation(); UmbuzoApp.deleteChat(${chat.id})">✕</button>
                    </div>
                `).join('');
            } else {
                chatListEl.innerHTML = `
                    <div class="nav-item" style="cursor: default; color: var(--text-tertiary);">
                        No chats yet. Start a conversation!
                    </div>
                `;
            }
        } catch (e) {
            console.error('Failed to load chat list:', e);
        }
    },

    /**
     * Open a chat
     */
    openChat(chatId) {
        this.currentChatId = chatId;
        window.history.pushState({}, '', `/?chat_id=${chatId}`);
        this.loadChat(chatId);

        const titleEl = document.querySelector('.top-nav-title');
        if (titleEl) {
            const chatItem = document.querySelector(`.chat-list-item[data-chat-id="${chatId}"]`);
            if (chatItem) {
                titleEl.textContent = chatItem.querySelector('.chat-title').textContent;
            }
        }

        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            document.querySelector('.sidebar')?.classList.remove('open');
        }
    },

    /**
     * Delete a chat
     */
    async deleteChat(chatId) {
        if (!confirm('Are you sure you want to delete this chat?')) return;

        try {
            await UmbuzoAPI.deleteChat(chatId);

            if (this.currentChatId === chatId) {
                this.currentChatId = null;
                window.history.pushState({}, '', '/');
                this.showWelcomeScreen();

                const titleEl = document.querySelector('.top-nav-title');
                if (titleEl) titleEl.textContent = 'Chat';
            }

            await this.loadChatList();
            this.showToast('Chat deleted', 'success');
        } catch (e) {
            this.showToast('Failed to delete chat: ' + e.message, 'error');
        }
    },

    /**
     * Handle file upload - open to all
     */
    async handleFileUpload() {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = true;
        input.accept = '.txt,.pdf,.doc,.docx,.png,.jpg,.jpeg,.gif,.csv,.json';

        input.onchange = async (e) => {
            const files = Array.from(e.target.files);
            if (files.length === 0) return;

            for (const file of files) {
                try {
                    const result = await UmbuzoAPI.uploadFile(file);
                    this.uploadedFiles.push(result);
                    this.showToast(`Uploaded: ${file.name}`, 'success');
                } catch (err) {
                    this.showToast(`Failed to upload ${file.name}: ${err.message}`, 'error');
                }
            }
        };

        input.click();
    },

    /**
     * Handle voice mode
     */
    handleVoiceMode() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            this.showToast('Voice mode is not supported in your browser', 'error');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = true;
        recognition.continuous = false;

        const voiceBtn = document.querySelector('[data-tool="voice"]');
        if (voiceBtn) {
            voiceBtn.classList.toggle('active');
        }

        if (voiceBtn?.classList.contains('active')) {
            recognition.start();

            recognition.onresult = (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0].transcript)
                    .join('');

                const input = document.querySelector('.prompt-input');
                if (input) {
                    input.value = transcript;
                    input.dispatchEvent(new Event('input'));
                }
            };

            recognition.onend = () => {
                voiceBtn?.classList.remove('active');
                const input = document.querySelector('.prompt-input');
                if (input && input.value.trim()) {
                    this.handleSendMessage();
                }
            };

            recognition.onerror = () => {
                voiceBtn?.classList.remove('active');
                this.showToast('Voice recognition failed', 'error');
            };
        }
    },

    /**
     * Render GPTs directory
     */
    renderGpts(gpts) {
        const container = document.querySelector('.gpts-grid');
        if (!container) return;

        if (!gpts || gpts.length === 0) {
            container.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; color: var(--text-tertiary); padding: 40px;">
                    No tools available yet
                </div>
            `;
            return;
        }

        container.innerHTML = gpts.map(gpt => `
            <div class="gpt-card" onclick="UmbuzoApp.useGPT(${gpt.id}, '${gpt.name.replace(/'/g, "\\'")}')">
                <div class="gpt-icon">${gpt.icon}</div>
                <div class="gpt-name">${gpt.name}</div>
                <div class="gpt-description">${gpt.description}</div>
                <span class="gpt-category">${gpt.category}</span>
            </div>
        `).join('');
    },

    /**
     * Use a GPT tool
     */
    async useGPT(gptId, name) {
        try {
            const chat = await UmbuzoAPI.createChat(`Using ${name}`, this.currentModel);
            window.location.href = `/?chat_id=${chat.id}`;
        } catch (e) {
            this.showToast('Failed to create chat: ' + e.message, 'error');
        }
    },

    /**
     * Render chat history
     */
    renderHistory(chats) {
        const list = document.querySelector('.history-list');
        if (!list) return;

        if (!chats || chats.length === 0) {
            list.innerHTML = `
                <div style="text-align: center; color: var(--text-tertiary); padding: 60px 20px;">
                    <p style="font-size: 1.2rem; margin-bottom: 8px;">No chat history yet</p>
                    <p>Start a new conversation to see it here</p>
                    <button class="btn-primary" style="width: auto; margin-top: 20px; padding: 10px 24px;"
                            onclick="window.location.href='/'">
                        Start New Chat
                    </button>
                </div>
            `;
            return;
        }

        list.innerHTML = chats.map(chat => {
            const date = new Date(chat.created_at).toLocaleDateString('en-US', {
                month: 'short', day: 'numeric', year: 'numeric'
            });
            const time = new Date(chat.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            return `
                <div class="history-item" onclick="window.location.href='/?chat_id=${chat.id}'">
                    <div class="history-item-icon">💬</div>
                    <div class="history-item-content">
                        <div class="history-item-title">${chat.title}</div>
                        <div class="history-item-meta">
                            <span>${date}</span>
                            <span>${time}</span>
                            <span>${chat.message_count} messages</span>
                        </div>
                    </div>
                    <div class="history-item-actions">
                        <button class="delete-btn" onclick="event.stopPropagation(); UmbuzoApp.deleteChat(${chat.id})">🗑️</button>
                    </div>
                </div>
            `;
        }).join('');

        const searchInput = document.querySelector('.history-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                document.querySelectorAll('.history-item').forEach(item => {
                    const title = item.querySelector('.history-item-title')?.textContent?.toLowerCase() || '';
                    item.style.display = title.includes(query) ? 'flex' : 'none';
                });
            });
        }
    },

    /**
     * Handle logout
     */
    async handleLogout() {
        try {
            await UmbuzoAPI.logout();
        } catch (e) {
            localStorage.removeItem('umbuzo_token');
            localStorage.removeItem('umbuzo_user');
        }
        window.location.href = '/';
    },

    /**
     * Show loading state
     */
    showLoading(show) {
        const container = document.querySelector('.messages-container');
        if (!container) return;
        container.style.opacity = show ? '0.6' : '1';
    },

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.remove();
            if (container.children.length === 0) {
                container.remove();
            }
        }, 3000);
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => UmbuzoApp.init());

// Make app globally available
window.UmbuzoApp = UmbuzoApp;