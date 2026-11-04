// ============================================================
// MEDIEXPLAIN AI - PROFESSIONAL FRONTEND CONTROLLER
// Complete Implementation: Phases 1 through 8
// ============================================================

const BASE_API_URL = "http://127.0.0.1:8000/api";
const CHAT_API_URL = `${BASE_API_URL}/chat`;
const CONVERSATIONS_API_URL = `${BASE_API_URL}/conversations`;
const UPLOAD_DOC_URL = `${BASE_API_URL}/upload-doc`;
const EXPLAIN_REPORT_URL = `${BASE_API_URL}/explain-medical-report`;
const OCR_PRESCRIPTION_URL = `${BASE_API_URL}/ocr-prescription`;
const EXPLAIN_PRESCRIPTION_URL = `${BASE_API_URL}/explain-prescription`;
const TREATMENT_API_URL = `${BASE_API_URL}/treatment-compare`;
const EXPLAIN_SELECTION_URL = `${BASE_API_URL}/explain-selection`;
const REPORT_API_URL = `${BASE_API_URL}/generate-report`;
const DOCTOR_SUMMARY_URL = `${BASE_API_URL}/doctor-summary`;

// Global State
let isRequestInProgress = false;
let activeDocId = null;
let activeDocName = null;
let activeDocAnalysis = null;
let activePrescriptionId = null;
let activePrescriptionData = null;
let currentSpeechUtterance = null;
let speechRecognitionInstance = null;
let isRecordingVoice = false;

// Current Active Interaction Data for Reports & Modals
let lastActiveResponse = null;
let lastActiveQuestion = "";

// Text Selection State for Word Explanation (Phase 6)
let currentSelectedText = "";
let currentSelectedContext = "";

// Chat History Storage (Phase 1)
const CHAT_HISTORY_KEY = "mediexplain_chat_history_v3";
let activeChatId = null;
let medicalChats = [];


// ============================================================
// TOAST FEEDBACK HELPER
// ============================================================

function showToast(message, duration = 3000) {
    const toast = document.getElementById("copyToast");
    if (!toast) return;

    toast.textContent = message;
    toast.classList.remove("hidden");

    setTimeout(() => {
        toast.classList.add("hidden");
    }, duration);
}


// ============================================================
// ERROR BANNER HELPERS
// ============================================================

function clearMedicalError() {
    const existing = document.getElementById("mediexplainErrorBanner");
    if (existing) existing.remove();
}

function showMedicalError(title, message, retryNotice = "") {
    clearMedicalError();

    const host = document.querySelector(".chat-question-card") || document.body;
    const banner = document.createElement("div");
    banner.id = "mediexplainErrorBanner";
    banner.className = "medical-error-banner";
    banner.setAttribute("role", "alert");

    const safeTitle = String(title || "Notice").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const safeMsg = String(message || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>");
    const safeRetry = retryNotice
        ? `<div class="medical-error-retry">⏱️ Recommended Wait: ${String(retryNotice).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>`
        : "";

    banner.innerHTML = `
        <div class="medical-error-header">
            <span class="medical-error-icon">⚠️</span>
            <span>${safeTitle}</span>
        </div>
        <div class="medical-error-body">${safeMsg}</div>
        ${safeRetry}
    `;

    host.appendChild(banner);
}


// ============================================================
// MOBILE SIDEBAR TOGGLE (PHASE 8)
// ============================================================

function toggleSidebar(forceState) {
    const sidebar = document.getElementById("chatSidebar");
    const backdrop = document.getElementById("sidebarBackdrop");
    if (!sidebar) return;

    const isOpen = (forceState !== undefined) ? forceState : !sidebar.classList.contains("sidebar-open");
    if (isOpen) {
        sidebar.classList.add("sidebar-open");
        if (backdrop) backdrop.classList.remove("hidden");
    } else {
        sidebar.classList.remove("sidebar-open");
        if (backdrop) backdrop.classList.add("hidden");
    }
}


// ============================================================
// CHAT HISTORY STORAGE & MANAGEMENT (PHASE 1)
// ============================================================

function generateStableId(prefix = "conv") {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

function loadMedicalChats() {
    try {
        const saved = localStorage.getItem(CHAT_HISTORY_KEY);
        let list = saved ? JSON.parse(saved) : [];
        if (!Array.isArray(list)) list = [];

        // Deduplicate by ID and clean up empty duplicate conversations
        const uniqueMap = new Map();
        list.forEach(c => {
            if (!c || !c.id) return;
            if (!uniqueMap.has(c.id)) {
                uniqueMap.set(c.id, c);
            } else {
                const existing = uniqueMap.get(c.id);
                if ((c.messages || []).length > (existing.messages || []).length) {
                    uniqueMap.set(c.id, c);
                }
            }
        });

        const deduped = Array.from(uniqueMap.values());
        deduped.sort((a, b) => (b.updatedAt || b.createdAt || "").localeCompare(a.updatedAt || a.createdAt || ""));
        return deduped;
    } catch (e) {
        console.error("Unable to load chat history from localStorage:", e);
        return [];
    }
}

function saveMedicalChats() {
    try {
        // Filter out empty chats unless it's currently active and newly created
        const toSave = medicalChats.filter(c => {
            if (!c || !c.id) return false;
            if (c.messages && c.messages.length > 0) return true;
            return c.id === activeChatId;
        });
        localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(toSave));
        return true;
    } catch (e) {
        console.error("Unable to save chat history:", e);
        showToast("Storage notice: Could not write to local storage.");
        return false;
    }
}

async function syncConversationsWithBackend() {
    try {
        const resp = await fetch(CONVERSATIONS_API_URL);
        if (!resp.ok) return;
        const backendList = await resp.json();
        if (Array.isArray(backendList) && backendList.length) {
            const localMap = new Map(medicalChats.map(c => [c.id, c]));

            backendList.forEach(bc => {
                if (!bc || !bc.id) return;
                if (!localMap.has(bc.id)) {
                    // Only add conversations that have messages or are the active chat
                    if (bc.messages && bc.messages.length > 0) {
                        medicalChats.push(bc);
                    }
                } else {
                    const local = localMap.get(bc.id);
                    // Never overwrite local messages with an empty array
                    if (bc.messages && bc.messages.length > 0) {
                        if ((bc.updatedAt || "") >= (local.updatedAt || "") || (!local.messages || local.messages.length === 0)) {
                            Object.assign(local, bc);
                        }
                    }
                }
            });

            // Deduplicate conversations across title and first question to avoid duplicates
            const seenExchanges = new Set();
            const cleanedList = [];
            medicalChats.forEach(c => {
                const firstQ = c.messages && c.messages[0] ? (c.messages[0].question || "").trim().toLowerCase() : "";
                const key = firstQ ? `${c.title}:::${firstQ}` : c.id;
                if (!seenExchanges.has(key)) {
                    seenExchanges.add(key);
                    cleanedList.push(c);
                } else {
                    // If an existing item had 0 messages and this one has messages, replace
                    const existingIdx = cleanedList.findIndex(x => (x.title === c.title && (x.messages && x.messages[0]?.question?.trim().toLowerCase() === firstQ)));
                    if (existingIdx !== -1 && (!cleanedList[existingIdx].messages || cleanedList[existingIdx].messages.length === 0) && (c.messages && c.messages.length > 0)) {
                        cleanedList[existingIdx] = c;
                    }
                }
            });

            medicalChats = cleanedList;
            medicalChats.sort((a, b) => (b.updatedAt || b.createdAt || "").localeCompare(a.updatedAt || a.createdAt || ""));
            saveMedicalChats();
            renderMedicalChatHistory();
        }
    } catch (err) {
        console.log("Backend conversations sync bypassed (offline or not reachable).");
    }
}

function getOrCreateActiveChat(initialQuestion = "Medical Question") {
    if (!activeChatId) {
        activeChatId = generateStableId("conv");
    }

    let chat = medicalChats.find(c => c.id === activeChatId);
    if (!chat) {
        const shortTitle = initialQuestion.slice(0, 42) + (initialQuestion.length > 42 ? "..." : "");
        chat = {
            id: activeChatId,
            title: shortTitle,
            messages: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };
        medicalChats.unshift(chat);
        saveMedicalChats();
        renderMedicalChatHistory();
    }
    return chat;
}

function renderMedicalChatHistory() {
    const list = document.getElementById("chatHistoryList");
    if (!list) return;

    list.innerHTML = "";

    // Filter to only display chats that have messages or are currently active
    const displayChats = medicalChats.filter(c => (c.messages && c.messages.length > 0) || c.id === activeChatId);

    if (displayChats.length === 0) {
        list.innerHTML = "<p class='chat-history-empty'>No saved conversations yet.</p>";
        return;
    }

    displayChats.forEach(chat => {
        const item = document.createElement("div");
        item.className = "chat-history-item";
        item.setAttribute("data-chat-id", chat.id);
        if (chat.id === activeChatId) item.classList.add("active");

        const openBtn = document.createElement("button");
        openBtn.type = "button";
        openBtn.className = "chat-history-open";
        openBtn.textContent = chat.title || "Medical Question";
        openBtn.title = chat.title || "Medical Question";
        openBtn.onclick = () => openMedicalChat(chat.id);

        const actions = document.createElement("div");
        actions.className = "chat-history-actions";

        const renameBtn = document.createElement("button");
        renameBtn.type = "button";
        renameBtn.innerHTML = "✎";
        renameBtn.title = "Rename conversation";
        renameBtn.onclick = (e) => {
            e.stopPropagation();
            renameMedicalChat(chat.id);
        };

        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.innerHTML = "×";
        delBtn.title = "Delete conversation";
        delBtn.onclick = (e) => {
            e.stopPropagation();
            deleteMedicalChat(chat.id);
        };

        actions.appendChild(renameBtn);
        actions.appendChild(delBtn);
        item.appendChild(openBtn);
        item.appendChild(actions);
        list.appendChild(item);
    });
}

async function openMedicalChat(chatId) {
    if (!chatId) return;
    let chat = medicalChats.find(c => c.id === chatId);

    // If not found locally or empty, attempt fetch from backend
    if (!chat || !chat.messages || chat.messages.length === 0) {
        try {
            const resp = await fetch(`${CONVERSATIONS_API_URL}/${chatId}`);
            if (resp.ok) {
                const fetchedChat = await resp.json();
                if (fetchedChat && fetchedChat.id) {
                    if (chat) {
                        Object.assign(chat, fetchedChat);
                    } else {
                        medicalChats.unshift(fetchedChat);
                        chat = fetchedChat;
                    }
                    saveMedicalChats();
                }
            }
        } catch (_) {}
    }

    if (!chat) return;

    activeChatId = chat.id;
    clearMedicalChatDisplay();
    toggleSidebar(false);

    const welcome = document.getElementById("chatWelcome");
    const thread = document.getElementById("conversationThread");
    const response = document.getElementById("responseSection");
    const answer = document.getElementById("answer");
    const sources = document.getElementById("sources");
    const confidence = document.getElementById("confidence");
    const safetyLevel = document.getElementById("safetyLevel");
    const safetyMessage = document.getElementById("safetyMessage");
    const questionInput = document.getElementById("question");

    if (chat.messages && chat.messages.length > 0) {
        if (welcome) welcome.style.display = "none";

        // Render full conversation thread
        renderConversationThread(chat);

        // Render details of the latest message in the response card
        const lastMsg = chat.messages[chat.messages.length - 1];
        lastActiveResponse = lastMsg;
        lastActiveQuestion = lastMsg.question;

        if (answer) answer.innerHTML = formatMedicalAnswer(lastMsg.answer || "No explanation recorded.");
        if (confidence) {
            confidence.textContent = lastMsg.confidence || "Normal";
            confidence.className = `confidence-${String(lastMsg.confidence || "normal").toLowerCase()}`;
        }
        if (safetyLevel) {
            safetyLevel.textContent = lastMsg.safetyLevel || "Normal";
            safetyLevel.className = `safety-${String(lastMsg.safetyLevel || "normal").toLowerCase()}`;
        }
        if (safetyMessage) safetyMessage.textContent = lastMsg.safetyMessage || "";

        applySafetyCardStyling(lastMsg.safetyLevel);

        renderTerminology(lastMsg.terminology || []);
        renderEvidenceExplorer(lastMsg.evidence_explorer || []);
        if (sources) renderSources(sources, lastMsg.sources || []);

        if (response) {
            response.classList.remove("hidden");
        }
    } else {
        if (welcome) welcome.style.display = "";
        if (thread) thread.classList.add("hidden");
        if (response) response.classList.add("hidden");
    }

    if (questionInput) questionInput.focus();
    renderMedicalChatHistory();
}

function renderConversationThread(chat) {
    const thread = document.getElementById("conversationThread");
    if (!thread) return;

    if (!chat || !chat.messages || chat.messages.length === 0) {
        thread.classList.add("hidden");
        thread.innerHTML = "";
        return;
    }

    thread.classList.remove("hidden");
    thread.innerHTML = `
        <div class="thread-header">
            <span>💬 Conversation History (${chat.messages.length} exchange${chat.messages.length > 1 ? "s" : ""})</span>
            <span class="thread-title-badge">${chat.title}</span>
        </div>
    `;

    chat.messages.forEach((msg, idx) => {
        const exchangeEl = document.createElement("div");
        exchangeEl.className = "thread-exchange-card";

        const qSafe = String(msg.question || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        const aFormatted = formatMedicalAnswer(msg.answer || "");
        const confBadge = `<span class="thread-badge badge-conf-${String(msg.confidence || "normal").toLowerCase()}">${msg.confidence || "Normal"}</span>`;
        const safetyBadge = `<span class="thread-badge badge-safety-${String(msg.safetyLevel || "normal").toLowerCase()}">${msg.safetyLevel || "Normal"}</span>`;

        exchangeEl.innerHTML = `
            <div class="thread-user-row">
                <span class="thread-role-pill user-pill">You (Turn ${idx + 1})</span>
                <span class="thread-question-text">${qSafe}</span>
            </div>
            <div class="thread-assistant-row selectable-answer-area">
                <div class="thread-assistant-header">
                    <span class="thread-role-pill ai-pill">MediExplain AI</span>
                    <div class="thread-badges-group">
                        ${confBadge}
                        ${safetyBadge}
                    </div>
                </div>
                <div class="thread-answer-text">${aFormatted}</div>
                ${msg.sources && msg.sources.length ? `
                    <div class="thread-citations-mini">
                        📚 Sources: ${msg.sources.map(s => `<span>${s.title} (${s.sources})</span>`).join(", ")}
                    </div>
                ` : ""}
            </div>
        `;

        thread.appendChild(exchangeEl);
    });
}

function renderConversationThreadWithPendingQuestion(chat, pendingQuestion) {
    const thread = document.getElementById("conversationThread");
    if (!thread) return;

    thread.classList.remove("hidden");
    const existingCount = (chat && chat.messages) ? chat.messages.length : 0;

    let html = `
        <div class="thread-header">
            <span>💬 Active Consultation</span>
            <span class="thread-title-badge">${chat ? chat.title : "Medical Chat"}</span>
        </div>
    `;

    if (chat && chat.messages) {
        chat.messages.forEach((msg, idx) => {
            const qSafe = String(msg.question || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            const aFormatted = formatMedicalAnswer(msg.answer || "");
            const confBadge = `<span class="thread-badge badge-conf-${String(msg.confidence || "normal").toLowerCase()}">${msg.confidence || "Normal"}</span>`;
            const safetyBadge = `<span class="thread-badge badge-safety-${String(msg.safetyLevel || "normal").toLowerCase()}">${msg.safetyLevel || "Normal"}</span>`;

            html += `
                <div class="thread-exchange-card">
                    <div class="thread-user-row">
                        <span class="thread-role-pill user-pill">You (Turn ${idx + 1})</span>
                        <span class="thread-question-text">${qSafe}</span>
                    </div>
                    <div class="thread-assistant-row selectable-answer-area">
                        <div class="thread-assistant-header">
                            <span class="thread-role-pill ai-pill">MediExplain AI</span>
                            <div class="thread-badges-group">
                                ${confBadge}
                                ${safetyBadge}
                            </div>
                        </div>
                        <div class="thread-answer-text">${aFormatted}</div>
                    </div>
                </div>
            `;
        });
    }

    const pendingSafe = String(pendingQuestion || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    html += `
        <div class="thread-exchange-card pending-exchange" id="pendingExchangeCard">
            <div class="thread-user-row">
                <span class="thread-role-pill user-pill">You (Turn ${existingCount + 1})</span>
                <span class="thread-question-text">${pendingSafe}</span>
            </div>
            <div class="thread-assistant-row">
                <div class="thread-assistant-header">
                    <span class="thread-role-pill ai-pill">MediExplain AI</span>
                    <span class="thread-badge badge-conf-normal">Analyzing clinical evidence...</span>
                </div>
                <div class="thread-loading-pulse">
                    <span>Searching verified medical knowledge base and generating explanation...</span>
                </div>
            </div>
        </div>
    `;

    thread.innerHTML = html;
    thread.scrollIntoView({ behavior: "smooth", block: "start" });
}
        
function renameMedicalChat(chatId) {
    const chat = medicalChats.find(c => c.id === chatId);
    if (!chat) return;

    const newTitle = prompt("Enter a new title for this medical conversation:", chat.title);
    if (newTitle === null) return;

    const cleaned = newTitle.trim();
    if (!cleaned) {
        showToast("Title cannot be empty.");
        return;
    }

    chat.title = cleaned.slice(0, 75);
    chat.updatedAt = new Date().toISOString();
    saveMedicalChats();
    renderMedicalChatHistory();

    // Sync rename to backend
    fetch(`${CONVERSATIONS_API_URL}/${chat.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: chat.title })
    }).catch(() => {});

    showToast("Conversation renamed.");
}

function deleteMedicalChat(chatId) {
    const chat = medicalChats.find(c => c.id === chatId);
    if (!chat) return;

    if (!confirm(`Delete conversation "${chat.title}"?`)) return;

    medicalChats = medicalChats.filter(c => c.id !== chatId);
    saveMedicalChats();

    // Sync delete to backend
    fetch(`${CONVERSATIONS_API_URL}/${chatId}`, {
        method: "DELETE"
    }).catch(() => {});

    if (activeChatId === chatId) {
        if (medicalChats.length > 0) {
            openMedicalChat(medicalChats[0].id);
        } else {
            startNewMedicalChat();
        }
    } else {
        renderMedicalChatHistory();
    }

    showToast("Conversation deleted.");
}

function startNewMedicalChat() {
    activeChatId = null;
    clearMedicalChatDisplay();
    toggleSidebar(false);

    const welcome = document.getElementById("chatWelcome");
    if (welcome) welcome.style.display = "";

    const thread = document.getElementById("conversationThread");
    if (thread) {
        thread.classList.add("hidden");
        thread.innerHTML = "";
    }

    const questionInput = document.getElementById("question");
    if (questionInput) {
        questionInput.value = "";
        questionInput.focus();
    }

    renderMedicalChatHistory();
    showToast("Started a new chat session.");
}

function clearMedicalChatDisplay() {
    const question = document.getElementById("question");
    const answer = document.getElementById("answer");
    const sources = document.getElementById("sources");
    const response = document.getElementById("responseSection");
    const loading = document.getElementById("loading");
    const confidence = document.getElementById("confidence");
    const safetyLevel = document.getElementById("safetyLevel");
    const safetyMessage = document.getElementById("safetyMessage");
    const explorer = document.getElementById("evidenceExplorerCard");
    const terms = document.getElementById("terminologyCard");

    if (question) question.value = "";
    if (answer) answer.innerHTML = "";
    if (sources) sources.innerHTML = "";
    if (response) response.classList.add("hidden");
    if (loading) loading.classList.add("hidden");
    if (confidence) confidence.textContent = "-";
    if (safetyLevel) safetyLevel.textContent = "-";
    if (safetyMessage) safetyMessage.textContent = "";
    if (explorer) explorer.classList.add("hidden");
    if (terms) terms.classList.add("hidden");

    removeAttachedDoc();
    removeAttachedPrescription();
    clearMedicalError();
    stopSpeech();
    closeWordExplanation();
}


// ============================================================
// MAIN QUESTION SUBMISSION (ASK MEDIEXPLAIN - PHASES 1 & 5)
// ============================================================

async function askMediExplain() {
    if (isRequestInProgress) return;

    const questionInput = document.getElementById("question");
    const languageInput = document.getElementById("language");
    const answer = document.getElementById("answer");
    const sources = document.getElementById("sources");
    const confidence = document.getElementById("confidence");
    const safetyLevel = document.getElementById("safetyLevel");
    const safetyMessage = document.getElementById("safetyMessage");
    const responseSection = document.getElementById("responseSection");
    const loading = document.getElementById("loading");
    const askButton = document.getElementById("askButton");
    const welcome = document.getElementById("chatWelcome");

    const question = questionInput.value.trim();
    const language = languageInput.value || "English";

    // 1. Validation
    if (!question) {
        showToast("Please enter a medical question.");
        questionInput.focus();
        return;
    }

    clearMedicalError();
    closeWordExplanation();
    isRequestInProgress = true;
    stopSpeech();

    // Ensure active conversation has a stable ID upfront
    if (!activeChatId) {
        activeChatId = generateStableId("conv");
    }
    const currentChat = getOrCreateActiveChat(question);

    // 2. Display the user's question immediately in the main chat area
    if (welcome) welcome.style.display = "none";
    renderConversationThreadWithPendingQuestion(currentChat, question);

    // 3. Show loading indicator and update button
    if (loading) loading.classList.remove("hidden");
    if (responseSection) responseSection.classList.add("hidden");

    if (askButton) {
        askButton.disabled = true;
        askButton.innerHTML = "<span>Analyzing evidence...</span> <span>⏳</span>";
    }

    // Reset textarea immediately so user sees submission registered
    questionInput.value = "";
    const count = document.getElementById("questionCount");
    if (count) count.textContent = "0/600";

    // Prepare multi-turn history from active chat (Phase 5)
    const historyPayload = [];
    if (currentChat && currentChat.messages && currentChat.messages.length) {
        const recent = currentChat.messages.slice(-4);
        recent.forEach(m => {
            if (m.question) historyPayload.push({ role: "user", content: m.question });
            if (m.answer) historyPayload.push({ role: "assistant", content: m.answer.slice(0, 350) });
        });
    }

    try {
        // 4. Send the question to the existing FastAPI backend
        const response = await fetch(CHAT_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: question,
                conversation_id: activeChatId,
                language: language,
                history: historyPayload.length ? historyPayload : null,
                doc_id: activeDocId || null,
                prescription_id: activePrescriptionId || null
            })
        });

        // 5. Handle HTTP Errors (with clear user retry without silent saves)
        if (!response.ok) {
            let errorPayload = null;
            try {
                errorPayload = await response.json();
            } catch (_) {
                try {
                    errorPayload = { detail: await response.text() };
                } catch (__) {
                    errorPayload = null;
                }
            }

            const detail = (errorPayload && errorPayload.detail) ? errorPayload.detail : {};
            const errorType = (typeof detail === "object" && detail !== null) ? (detail.error_type || "") : "";
            const detailMsg = (typeof detail === "object" && detail !== null) ? (detail.message || "") : String(detail || "");
            const retryAfter = (typeof detail === "object" && detail !== null) ? (detail.retry_after || "") : "";
            const isQuota = (typeof detail === "object" && detail !== null) ? Boolean(detail.is_quota) : false;

            let userTitle = "MediExplain Notice";
            let userMessage = detailMsg || `Server returned status ${response.status}.`;
            let retryNotice = retryAfter;

            if (response.status === 429) {
                if (isQuota || errorType === "quota_exceeded") {
                    userTitle = "AI Service Quota Limit Reached";
                    userMessage = detailMsg || "The configured OpenAI account has reached its billing quota. The application remains fully functional for health terminology, safety checks, and reference search.";
                    retryNotice = "";
                } else {
                    userTitle = "Service Temporarily Busy (Rate Limit)";
                    userMessage = detailMsg || "The AI model has reached its token rate limit. Please try again shortly.";
                }
            } else if (response.status === 401) {
                userTitle = "Authentication Required";
                userMessage = detailMsg || "OpenAI API key validation failed in the backend configuration.";
            } else if (response.status === 502 || errorType === "generator_error") {
                userTitle = "Medical Generator Service Error";
                userMessage = detailMsg || "An error occurred with the AI service. Please retry in a moment.";
            } else if (response.status >= 500) {
                userTitle = `Server Error (${response.status})`;
                userMessage = detailMsg || "An internal error occurred on the MediExplain AI backend.";
            }

            showMedicalError(userTitle, userMessage, retryNotice);

            // Re-render chat thread without pending ghost message and restore question in input for easy retry
            renderConversationThread(currentChat);
            questionInput.value = question;
            if (count) count.textContent = `${question.length}/600`;
            return;
        }

        // 6. Receive the generated answer
        const data = await response.json();
        lastActiveResponse = data;
        lastActiveQuestion = question;

        if (data.conversation_id && data.conversation_id !== activeChatId) {
            activeChatId = data.conversation_id;
            currentChat.id = data.conversation_id;
        }

        // 7. Save completed exchange to active conversation exactly once
        const turnRecord = {
            id: generateStableId("msg"),
            question: question,
            answer: data.answer || "No explanation received.",
            confidence: data.confidence || "Normal",
            safetyLevel: data.safety_level || "Normal",
            safetyMessage: data.safety_message || "",
            sources: data.sources || [],
            evidence_explorer: data.evidence_explorer || [],
            terminology: data.terminology || [],
            language: language,
            doc_id: activeDocId,
            doc_name: activeDocName,
            timestamp: new Date().toISOString()
        };

        // Prevent duplicate turns
        const isDuplicateTurn = currentChat.messages.some(m => m.question === question && m.answer === turnRecord.answer);
        if (!isDuplicateTurn) {
            currentChat.messages.push(turnRecord);
        }

        if (currentChat.messages.length === 1) {
            currentChat.title = question.slice(0, 42) + (question.length > 42 ? "..." : "");
        }
        currentChat.updatedAt = new Date().toISOString();

        saveMedicalChats();
        renderMedicalChatHistory();

        // 8. Display the answer in the main chat area (conversation thread + response card)
        renderConversationThread(currentChat);

        if (answer) answer.innerHTML = formatMedicalAnswer(data.answer || "No explanation received.");

        if (confidence) {
            confidence.textContent = data.confidence || "Normal";
            confidence.className = `confidence-${String(data.confidence || "normal").toLowerCase()}`;
        }

        if (safetyLevel) {
            safetyLevel.textContent = data.safety_level || "Normal";
            safetyLevel.className = `safety-${String(data.safety_level || "normal").toLowerCase()}`;
        }

        if (safetyMessage) safetyMessage.textContent = data.safety_message || "This information is for educational purposes only.";
        applySafetyCardStyling(data.safety_level);

        renderTerminology(data.terminology || []);
        renderEvidenceExplorer(data.evidence_explorer || []);
        if (sources) renderSources(sources, data.sources || []);

        if (responseSection) {
            responseSection.classList.remove("hidden");
            responseSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }

    } catch (error) {
        console.error("MediExplain API Error:", error);
        const isNetwork = (error instanceof TypeError) || !window.navigator.onLine || String(error.message || "").toLowerCase().includes("failed to fetch");
        if (isNetwork) {
            showMedicalError(
                "Connection Notice",
                "Could not reach the MediExplain AI backend (http://127.0.0.1:8000). Please verify that the FastAPI server is running."
            );
        } else {
            showMedicalError(
                "Processing Error",
                error.message || "An unexpected error occurred while processing your request."
            );
        }

        // Restore thread and question for retry
        renderConversationThread(currentChat);
        questionInput.value = question;
        if (count) count.textContent = `${question.length}/600`;

    } finally {
        if (loading) loading.classList.add("hidden");
        if (askButton) {
            askButton.disabled = false;
            askButton.innerHTML = "<span>Ask MediExplain</span> <span>→</span>";
        }
        isRequestInProgress = false;
    }
}

function applySafetyCardStyling(levelStr) {
    const safetyCard = document.querySelector(".safety-card");
    if (!safetyCard) return;
    safetyCard.classList.remove("safety-emergency", "safety-warning", "safety-normal");
    const level = String(levelStr || "Normal").toLowerCase();
    if (level === "emergency") safetyCard.classList.add("safety-emergency");
    else if (level === "warning") safetyCard.classList.add("safety-warning");
    else safetyCard.classList.add("safety-normal");
}


// ============================================================
// FORMAT MEDICAL ANSWER
// ============================================================

function formatMedicalAnswer(text) {
    if (!text) return "No answer received.";

    let safe = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // Markdown Bold
    safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    // Headers
    safe = safe.replace(/^### (.*$)/gim, "<h4 class='answer-h4'>$1</h4>");
    safe = safe.replace(/^## (.*$)/gim, "<h3 class='answer-h3'>$1</h3>");

    // Line breaks
    safe = safe.replace(/\n/g, "<br>");

    // Bullet points
    safe = safe.replace(/(^|<br>)\s*[-•*]\s+/g, "$1<span class='answer-bullet'>•</span> ");

    return safe;
}


// ============================================================
// RENDER CITATIONS & EVIDENCE EXPLORER & TERMINOLOGY
// ============================================================

function renderSources(container, sourceList) {
    if (!container) return;
    container.innerHTML = "";

    if (!sourceList || sourceList.length === 0) {
        container.innerHTML = `
            <div class="no-sources">
                <span>📚</span>
                <p>No external citations were required or returned for this response.</p>
            </div>
        `;
        return;
    }

    sourceList.forEach(src => {
        const item = document.createElement("div");
        item.className = "source-item";

        const icon = document.createElement("div");
        icon.className = "source-icon";
        icon.textContent = "📖";

        const content = document.createElement("div");
        content.className = "source-content";

        const title = document.createElement("div");
        title.className = "source-title";
        title.textContent = src.title || "Medical Reference";

        const meta = document.createElement("div");
        meta.className = "source-meta";
        const srcName = src.sources || "MedlinePlus";
        const cat = src.category || "Medical Reference";
        meta.textContent = `${srcName} • ${cat}`;

        content.appendChild(title);
        content.appendChild(meta);

        if (src.claims && src.claims.length) {
            const claimEl = document.createElement("div");
            claimEl.className = "source-claim-snippet";
            claimEl.innerHTML = `<strong>Supported point:</strong> "${src.claims[0]}"`;
            content.appendChild(claimEl);
        }

        if (src.url) {
            const link = document.createElement("a");
            link.className = "source-link";
            link.href = src.url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = "View source →";
            content.appendChild(link);
        }

        item.appendChild(icon);
        item.appendChild(content);
        container.appendChild(item);
    });
}

function renderEvidenceExplorer(evidenceList) {
    const explorerCard = document.getElementById("evidenceExplorerCard");
    const container = document.getElementById("evidenceList");
    if (!explorerCard || !container) return;

    if (!evidenceList || !evidenceList.length) {
        explorerCard.classList.add("hidden");
        return;
    }

    explorerCard.classList.remove("hidden");
    container.innerHTML = "";

    evidenceList.forEach((ev, idx) => {
        const item = document.createElement("div");
        item.className = "evidence-item";

        const scorePercent = Math.round((ev.relevance_score || 0) * 100);
        const sourceLabel = ev.source ? ` (${ev.source})` : "";

        item.innerHTML = `
            <div class="evidence-item-top">
                <span class="evidence-title">Passage ${idx + 1}: ${ev.title}${sourceLabel}</span>
                <span class="evidence-score-badge">Relevance: ${scorePercent}%</span>
            </div>
            <div class="evidence-excerpt">"${ev.excerpt || "No excerpt extracted."}"</div>
            ${ev.url ? `<a href="${ev.url}" target="_blank" rel="noopener noreferrer" class="evidence-source-link">Source link: ${ev.title} →</a>` : ""}
        `;

        container.appendChild(item);
    });
}

function renderTerminology(terms) {
    const termCard = document.getElementById("terminologyCard");
    const list = document.getElementById("terminologyList");
    if (!termCard || !list) return;

    if (!terms || !terms.length) {
        termCard.classList.add("hidden");
        return;
    }

    termCard.classList.remove("hidden");
    list.innerHTML = "";

    terms.forEach(t => {
        const item = document.createElement("div");
        item.className = "terminology-item";

        item.innerHTML = `
            <div class="term-title">${t.term}</div>
            <div class="term-definition"><strong>Simple definition:</strong> ${t.definition}</div>
            <div class="term-detail"><strong>Why it matters:</strong> ${t.importance}</div>
            ${t.example ? `<div class="term-detail"><strong>Example:</strong> ${t.example}</div>` : ""}
        `;

        list.appendChild(item);
    });
}


// ============================================================
// PDF MEDICAL REPORT UPLOAD & EXPLANATION (PHASE 2)
// ============================================================

async function handlePdfUpload(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showToast("Only PDF medical documents are supported.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showToast("PDF file exceeds the maximum 10 MB limit.");
        return;
    }

    const badgeContainer = document.getElementById("attachedDocContainer");
    badgeContainer.classList.remove("hidden");
    badgeContainer.innerHTML = `<span class="attached-doc-badge">⏳ Processing & extracting ${file.name}...</span>`;

    try {
        const base64Data = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const res = reader.result;
                const base64 = (typeof res === "string" && res.includes(",")) ? res.split(",")[1] : res;
                resolve(base64);
            };
            reader.onerror = () => reject(new Error("Failed to read file"));
            reader.readAsDataURL(file);
        });

        const resp = await fetch(UPLOAD_DOC_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: file.name,
                file_base64: base64Data
            })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail ? (err.detail.message || JSON.stringify(err.detail)) : `Upload failed (${resp.status})`);
        }

        const data = await resp.json();
        activeDocId = data.doc_id;
        activeDocName = data.filename;
        activeDocAnalysis = data.analysis;

        const scanBadge = data.is_scanned ? " [OCR Scanned]" : "";

        badgeContainer.innerHTML = `
            <div class="attached-doc-badge">
                <span>📄 ${data.filename} (${data.pages} page${data.pages > 1 ? "s" : ""}${scanBadge})</span>
                <button type="button" class="attached-view-btn" onclick="openPdfReportModal()">View Report Explanation</button>
                <button type="button" class="attached-remove-btn" onclick="removeAttachedDoc()" title="Remove document">×</button>
            </div>
        `;

        showToast(`Document parsed! You can ask follow-up questions or view the structured report.`);
    } catch (err) {
        console.error("PDF upload error:", err);
        showToast(`PDF upload failed: ${err.message}`);
        badgeContainer.classList.add("hidden");
        badgeContainer.innerHTML = "";
    }
}

function removeAttachedDoc() {
    activeDocId = null;
    activeDocName = null;
    activeDocAnalysis = null;
    const badgeContainer = document.getElementById("attachedDocContainer");
    if (badgeContainer) {
        badgeContainer.classList.add("hidden");
        badgeContainer.innerHTML = "";
    }
    const input = document.getElementById("pdfFileInput");
    if (input) input.value = "";
}

async function openPdfReportModal() {
    if (!activeDocId) {
        showToast("Please upload a PDF medical report first.");
        return;
    }

    const modal = document.getElementById("pdfReportModal");
    const titleEl = document.getElementById("pdfModalReportTitle");
    const bodyEl = document.getElementById("pdfReportModalBody");

    modal.classList.remove("hidden");
    bodyEl.innerHTML = "<p>Analyzing structured laboratory report...</p>";

    try {
        const resp = await fetch(EXPLAIN_REPORT_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ doc_id: activeDocId, language: document.getElementById("language").value || "English" })
        });

        if (!resp.ok) throw new Error("Could not retrieve report explanation.");
        const data = await resp.json();

        titleEl.textContent = `Report Analysis: ${data.filename}`;

        // Patient details
        let patientHTML = "";
        const pInfo = data.patient_info || {};
        if (Object.keys(pInfo).length > 0) {
            patientHTML = `
                <div class="report-meta-box">
                    ${pInfo.age ? `<span><strong>Age:</strong> ${pInfo.age}</span>` : ""}
                    ${pInfo.gender ? `<span><strong>Gender:</strong> ${pInfo.gender}</span>` : ""}
                    ${pInfo.report_date ? `<span><strong>Date:</strong> ${pInfo.report_date}</span>` : ""}
                    <span><strong>Pages:</strong> ${data.pages} ${data.is_scanned ? "(Scanned with OCR)" : "(Digital)"}</span>
                </div>
            `;
        }

        // Abnormal Findings
        let abnormalHTML = "";
        if (data.abnormal_findings && data.abnormal_findings.length) {
            abnormalHTML = `
                <div class="report-alert-card">
                    <h4>⚠️ Flagged / Out-of-Range Results (${data.abnormal_findings.length})</h4>
                    <p style="font-size:13px; color:#d6e6f5; margin-bottom:12px;">The following test results are outside standard reference intervals and warrant discussion with your physician:</p>
                    <div class="abnormal-list">
                        ${data.abnormal_findings.map(ab => `
                            <div class="abnormal-item">
                                <div><strong>${ab.test_name}</strong>: <span class="abnormal-val">${ab.value}</span> (Ref: ${ab.reference_range}) [Page ${ab.page_number}]</div>
                                <div class="abnormal-desc">${ab.significance}</div>
                            </div>
                        `).join("")}
                    </div>
                </div>
            `;
        }

        // Analyzed Tests Table
        let testsHTML = "";
        if (data.tests_analyzed && data.tests_analyzed.length) {
            testsHTML = `
                <div class="report-section">
                    <h3>Laboratory & Diagnostic Values Analyzed</h3>
                    <div class="treatment-table-wrapper">
                        <table class="treatment-table">
                            <thead>
                                <tr>
                                    <th>Test Name</th>
                                    <th>Observed Value</th>
                                    <th>Reference Range</th>
                                    <th>Status</th>
                                    <th>Plain-Language Significance</th>
                                    <th>Page</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${data.tests_analyzed.map(t => {
                                    const statusClass = t.status.includes("High") ? "status-high" : (t.status.includes("Low") ? "status-low" : "status-normal");
                                    return `
                                        <tr>
                                            <td><strong>${t.test_name}</strong></td>
                                            <td>${t.value} ${t.unit}</td>
                                            <td>${t.reference_range}</td>
                                            <td><span class="status-pill ${statusClass}">${t.status}</span></td>
                                            <td>${t.plain_explanation}</td>
                                            <td>Page ${t.page_number}</td>
                                        </tr>
                                    `;
                                }).join("")}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        } else {
            testsHTML = `
                <div class="report-section">
                    <p>No standard laboratory panel entries (like HbA1c, Glucose, Lipids, Creatinine) were recognized in this PDF. The full text has been indexed for follow-up questions.</p>
                </div>
            `;
        }

        // Recommendations & Follow-up
        let recsHTML = "";
        if (data.recommendations && data.recommendations.length) {
            recsHTML = `
                <div class="report-section">
                    <h3>Clinical Next Steps & Guidance</h3>
                    <ul>${data.recommendations.map(r => `<li>${r}</li>`).join("")}</ul>
                </div>
            `;
        }

        // Missing or Unclear Info
        let unclearHTML = "";
        if (data.unclear_or_missing_info && data.unclear_or_missing_info.length) {
            unclearHTML = `
                <div class="report-section" style="background:rgba(255,180,0,0.08); padding:12px; border-radius:8px;">
                    <strong>Notice on Unclear / Partially Stated Entries:</strong>
                    <p style="font-size:13px; margin-top:4px;">The following test headings were detected but specific numerical values could not be read clearly: ${data.unclear_or_missing_info.join(", ")}. Please inspect the physical paper report.</p>
                </div>
            `;
        }

        bodyEl.innerHTML = `
            ${patientHTML}
            ${abnormalHTML}
            ${testsHTML}
            ${recsHTML}
            ${unclearHTML}
            <div class="report-disclaimer-box">
                <strong>CLINICAL DISCLAIMER:</strong> ${data.disclaimer}
            </div>
        `;
    } catch (err) {
        bodyEl.innerHTML = `<p style="color:#ff8585;">Failed to load report explanation: ${err.message}</p>`;
    }
}


// ============================================================
// PRESCRIPTION IMAGE RECOGNITION & EXPLANATION (PHASE 3)
// ============================================================

async function handlePrescriptionUpload(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type.toLowerCase())) {
        showToast("Please upload a JPG, PNG, or WEBP prescription image.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showToast("Prescription image exceeds 10 MB limit.");
        return;
    }

    const modal = document.getElementById("prescriptionModal");
    const bodyEl = document.getElementById("prescriptionModalBody");

    modal.classList.remove("hidden");
    bodyEl.innerHTML = "<p>Preprocessing image (denoising, contrast enhancement) and performing OCR extraction...</p>";

    try {
        const base64Data = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const res = reader.result;
                const base64 = (typeof res === "string" && res.includes(",")) ? res.split(",")[1] : res;
                resolve(base64);
            };
            reader.onerror = () => reject(new Error("Failed to read image"));
            reader.readAsDataURL(file);
        });

        const resp = await fetch(OCR_PRESCRIPTION_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: file.name,
                image_base64: base64Data
            })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail ? (err.detail.message || JSON.stringify(err.detail)) : `OCR failed (${resp.status})`);
        }

        const data = await resp.json();
        activePrescriptionId = data.prescription_id;
        activePrescriptionData = data;

        renderPrescriptionReview(data, `data:${file.type};base64,${base64Data}`);

        // Update attachment badge in ask box
        const badgeContainer = document.getElementById("attachedImageContainer");
        badgeContainer.classList.remove("hidden");
        badgeContainer.innerHTML = `
            <div class="attached-doc-badge">
                <span>💊 Prescription: ${file.name} (${data.overall_readability})</span>
                <button type="button" class="attached-view-btn" onclick="document.getElementById('prescriptionModal').classList.remove('hidden')">Review Prescription</button>
                <button type="button" class="attached-remove-btn" onclick="removeAttachedPrescription()" title="Remove prescription">×</button>
            </div>
        `;

        showToast("Prescription extracted! Please review detected medicines before generating explanation.");
    } catch (err) {
        console.error("Prescription upload error:", err);
        bodyEl.innerHTML = `<p style="color:#ff8585;">Prescription recognition failed: ${err.message}</p>`;
    }
}

function renderPrescriptionReview(data, imageSrc) {
    const bodyEl = document.getElementById("prescriptionModalBody");
    if (!bodyEl) return;

    const medicines = data.medicines || [];
    const explanations = data.explanations || [];
    const explanationsMap = new Map();
    explanations.forEach(exp => {
        if (exp && exp.medicine_name) {
            explanationsMap.set(exp.medicine_name.toLowerCase(), exp);
        }
    });

    let medsHTML = "";
    if (medicines.length === 0) {
        medsHTML = `
            <div class="prescription-empty-notice">
                <p>⚠️ No standard medicine lines could be automatically identified from this image.</p>
                <p>You can add a medicine line below or consult your dispensing pharmacist to review the prescription.</p>
                <button type="button" class="tool-action-btn" onclick="addCustomPrescriptionRow()">+ Add Medicine</button>
            </div>
        `;
    } else {
        medsHTML = `
            <div class="prescription-review-list" id="prescriptionEditList">
                ${medicines.map((m, idx) => {
                    const isUnclear = m.clarity === "Unclear / Verification Required" || (m.readability_status && m.readability_status.includes("Unclear"));
                    const isProbable = m.clarity === "Probable Match (Needs Verification)" || (m.readability_status && m.readability_status.includes("Probable"));
                    const badgeClass = isUnclear ? "clarity-unclear" : (isProbable ? "clarity-medium" : "clarity-high");
                    const cardBorder = isUnclear ? "border: 1px solid rgba(255, 140, 0, 0.6);" : "";
                    
                    const exp = explanationsMap.get((m.medicine_name || "").toLowerCase()) || explanations[idx] || null;

                    return `
                        <div class="prescription-med-card" id="medCard_${idx}" style="${cardBorder}">
                            <div class="prescription-med-header">
                                <div>
                                    <span style="font-weight:700; color:#21e0d5; font-size:15px;">💊 ${m.medicine_name || "Prescription Item"}</span>
                                    ${m.matched_active_ingredient && m.matched_active_ingredient !== "Unverified" ? `<span style="font-size:12px; color:#8ab8d6; margin-left:8px;">(Active Ingredient: ${m.matched_active_ingredient})</span>` : ""}
                                </div>
                                <span class="clarity-badge ${badgeClass}">${m.readability_status || "Identified"}</span>
                            </div>

                            ${isUnclear ? `
                                <div class="prescription-unclear-banner" style="background:rgba(255, 140, 0, 0.12); padding:8px 12px; border-radius:6px; margin:8px 0; font-size:12px; color:#ffb347;">
                                    ⚠️ <strong>Unclear handwriting:</strong> The handwritten text for this item contains ambiguities (OCR text: <em>"${m.raw_text}"</em>). Please verify the exact medicine name and strength with your pharmacist.
                                </div>
                            ` : (isProbable ? `
                                <div class="prescription-probable-banner" style="background:rgba(33, 224, 213, 0.1); padding:6px 10px; border-radius:6px; margin:6px 0; font-size:12px; color:#21e0d5;">
                                    🔍 <strong>Probable match:</strong> Matched closest candidate from drug database (OCR text: <em>"${m.raw_text}"</em>).
                                </div>
                            ` : "")}

                            <div class="prescription-grid-inputs">
                                <div>
                                    <label>Medicine Name</label>
                                    <input type="text" id="medName_${idx}" value="${m.medicine_name || ""}" placeholder="e.g. Metformin">
                                </div>
                                <div>
                                    <label>Strength</label>
                                    <input type="text" id="medStrength_${idx}" value="${m.strength || ""}" placeholder="e.g. 500mg">
                                </div>
                                <div>
                                    <label>Dosage</label>
                                    <input type="text" id="medDosage_${idx}" value="${m.dosage || ""}" placeholder="e.g. 1 tablet">
                                </div>
                                <div>
                                    <label>Frequency</label>
                                    <input type="text" id="medFreq_${idx}" value="${m.frequency || ""}" placeholder="e.g. Twice daily (BID)">
                                </div>
                                <div>
                                    <label>Route</label>
                                    <input type="text" id="medRoute_${idx}" value="${m.route || "Oral"}" placeholder="Oral / IV / Inhalation">
                                </div>
                                <div>
                                    <label>Duration</label>
                                    <input type="text" id="medDur_${idx}" value="${m.duration || ""}" placeholder="e.g. 14 days">
                                </div>
                            </div>

                            <div style="margin-top:8px;">
                                <label>Special Instructions</label>
                                <input type="text" id="medInst_${idx}" value="${m.special_instructions || ""}" placeholder="e.g. After meals with water" style="width:100%;">
                            </div>

                            ${exp ? `
                                <div class="explained-med-item" style="margin-top:12px; background:rgba(10, 30, 50, 0.5); padding:12px; border-radius:8px; border-left:3px solid #21e0d5;">
                                    <div style="font-size:12px; text-transform:uppercase; color:#21e0d5; font-weight:700; margin-bottom:4px;">Patient Explanation & Guidance</div>
                                    <p style="margin:4px 0;"><strong>Primary Purpose:</strong> ${exp.purpose || "Clinical medication"}</p>
                                    <p style="margin:4px 0;"><strong>How to understand instructions:</strong> ${exp.instructions_guide || ""}</p>
                                    <p style="margin:4px 0;"><strong>Common Side Effects:</strong> ${Array.isArray(exp.common_side_effects) ? exp.common_side_effects.join(", ") : exp.common_side_effects}</p>
                                    <p style="margin:4px 0;"><strong>Important Precautions:</strong> ${exp.precautions || "Follow doctor instructions."}</p>
                                    <div style="margin-top:6px; font-size:11px; color:#8ab8d6;">📚 Source: ${exp.evidence_source || "MedlinePlus Drug Information"}</div>
                                </div>
                            ` : ""}
                        </div>
                    `;
                }).join("")}
            </div>
        `;
    }

    let interactionsHTML = "";
    if (data.interactions && data.interactions.length) {
        interactionsHTML = `
            <div class="prescription-interaction-box" style="margin-top:14px; background:rgba(255,180,0,0.08); padding:12px; border-radius:8px; border:1px solid rgba(255,180,0,0.3);">
                <h4 style="color:#ffb347; margin:0 0 6px 0;">🔍 Drug-Drug Interaction Review</h4>
                ${data.interactions.map(it => `
                    <div class="interaction-item" style="margin-bottom:6px; font-size:13px;">
                        <strong>${it.medicines}</strong>: ${it.evidence} <em style="color:#8ab8d6;">(${it.source})</em>
                    </div>
                `).join("")}
            </div>
        `;
    }

    let warningsHTML = "";
    if (data.safety_warnings && data.safety_warnings.length) {
        warningsHTML = `
            <div class="prescription-warnings-box" style="margin-top:12px;">
                ${data.safety_warnings.map(w => `<p style="color:#ffb347; font-size:13px; margin:4px 0;">${w}</p>`).join("")}
            </div>
        `;
    }

    bodyEl.innerHTML = `
        <div class="prescription-split-view">
            <div class="prescription-image-pane">
                <h4>Uploaded Prescription</h4>
                <div class="prescription-img-container">
                    <img src="${imageSrc}" alt="Uploaded Doctor Prescription" class="prescription-preview-img">
                </div>
                <div class="prescription-readability-meta">
                    <strong>Readability Status:</strong> ${data.overall_readability || "Medium"}
                </div>
                ${data.abbreviations_found && data.abbreviations_found.length ? `
                    <div class="prescription-abbr-box">
                        <strong>Identified Medical Abbreviations:</strong>
                        <ul>
                            ${data.abbreviations_found.map(a => `<li><strong>${a.abbreviation}:</strong> ${a.meaning}</li>`).join("")}
                        </ul>
                    </div>
                ` : ""}
                <div style="margin-top:12px;">
                    <button type="button" class="tool-action-btn" onclick="addCustomPrescriptionRow()" style="width:100%; font-size:13px;">+ Add / Adjust Medicine Row</button>
                </div>
            </div>

            <div class="prescription-data-pane">
                <h4>Recognized Prescription Medicines & Explanations</h4>
                <p style="font-size:13px; color:#c7ddef; margin-bottom:14px;">
                    Handwritten medicines and instructions were extracted using TrOCR & RapidOCR. Explanations are synthesized from verified medical sources. Please verify any flagged items.
                </p>

                ${medsHTML}
                ${interactionsHTML}
                ${warningsHTML}

                <div class="report-disclaimer-box" style="margin-top:16px;">
                    <strong>CLINICAL SAFETY NOTICE:</strong> ${data.disclaimer || "MediExplain AI does not modify or confirm prescriptions. Always verify unclear handwriting and dosage instructions with your dispensing pharmacist or prescribing doctor."}
                </div>

                <div class="prescription-action-row" style="margin-top:14px;">
                    <button type="button" class="chat-ask-button" onclick="submitPrescriptionForExplanation(${medicines.length})">
                        Re-check / Update Explanations →
                    </button>
                    <button type="button" class="tool-action-btn" onclick="closeModal('prescriptionModal'); askAboutPrescription();" style="margin-left:10px;">
                        💬 Discuss in Chat
                    </button>
                </div>

                <div id="prescriptionExplanationOutput" class="prescription-explanation-output hidden"></div>
            </div>
        </div>
    `;
}
function askAboutPrescription() {
    const qInput = document.getElementById("question");
    if (qInput) {
        qInput.value = "Can you summarize the precautions and how I should take the medicines on my uploaded prescription?";
        qInput.focus();
        askMediExplain();
    }
}

async function submitPrescriptionForExplanation(count) {
    const medicinesToExplain = [];
    for (let i = 0; i < count; i++) {
        const name = (document.getElementById(`medName_${i}`)?.value || "").trim();
        const strength = (document.getElementById(`medStrength_${i}`)?.value || "").trim();
        const dosage = (document.getElementById(`medDosage_${i}`)?.value || "").trim();
        const freq = (document.getElementById(`medFreq_${i}`)?.value || "").trim();
        const route = (document.getElementById(`medRoute_${i}`)?.value || "").trim();
        const dur = (document.getElementById(`medDur_${i}`)?.value || "").trim();
        const inst = (document.getElementById(`medInst_${i}`)?.value || "").trim();

        if (name) {
            medicinesToExplain.push({
                medicine_name: name,
                strength: strength,
                dosage: dosage,
                frequency: freq,
                route: route,
                duration: dur,
                special_instructions: inst
            });
        }
    }

    if (medicinesToExplain.length === 0) {
        showToast("Please provide at least one medicine name.");
        return;
    }

    const outputEl = document.getElementById("prescriptionExplanationOutput");
    outputEl.classList.remove("hidden");
    outputEl.innerHTML = "<p>Preparing medicine-by-medicine patient explanations and safety checks...</p>";

    try {
        const resp = await fetch(EXPLAIN_PRESCRIPTION_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prescription_id: activePrescriptionId,
                medicines: medicinesToExplain,
                language: document.getElementById("language").value || "English"
            })
        });

        if (!resp.ok) throw new Error("Could not generate explanation.");
        const data = await resp.json();

        let medsExplainHTML = data.medicines.map(m => `
            <div class="explained-med-item">
                <h5>💊 ${m.medicine_name} (${m.strength})</h5>
                <p><strong>Primary Purpose:</strong> ${m.purpose}</p>
                <p><strong>Prescription Guide:</strong> ${m.instructions_guide}</p>
                <p><strong>Common Side Effects:</strong> ${Array.isArray(m.common_side_effects) ? m.common_side_effects.join(", ") : m.common_side_effects}</p>
                <p><strong>Important Precautions:</strong> ${m.precautions}</p>
                ${m.unclear_notice ? `<p style="color:#ffb347;">⚠️ ${m.unclear_notice}</p>` : ""}
                <span class="explained-source-tag">Reference: ${m.evidence_source}</span>
            </div>
        `).join("");

        let interactionsHTML = "";
        if (data.interactions && data.interactions.length) {
            interactionsHTML = `
                <div class="prescription-interaction-box">
                    <h4>🔍 Drug-Drug Interaction Review</h4>
                    ${data.interactions.map(it => `
                        <div class="interaction-item">
                            <strong>${it.medicines}</strong>: ${it.evidence} <em>(${it.source})</em>
                        </div>
                    `).join("")}
                </div>
            `;
        }

        let warningsHTML = "";
        if (data.safety_warnings && data.safety_warnings.length) {
            warningsHTML = `
                <div class="prescription-warnings-box">
                    ${data.safety_warnings.map(w => `<p>${w}</p>`).join("")}
                </div>
            `;
        }

        outputEl.innerHTML = `
            <h4 style="margin-top:20px; color:#21e0d5;">Patient-Friendly Medicine Explanations</h4>
            ${medsExplainHTML}
            ${interactionsHTML}
            ${warningsHTML}
            <div class="report-disclaimer-box" style="margin-top:16px;">
                <strong>SAFETY WARNING:</strong> ${data.disclaimer}
            </div>
        `;
    } catch (err) {
        outputEl.innerHTML = `<p style="color:#ff8585;">Explanation failed: ${err.message}</p>`;
    }
}

function removeAttachedPrescription() {
    activePrescriptionId = null;
    activePrescriptionData = null;
    const badgeContainer = document.getElementById("attachedImageContainer");
    if (badgeContainer) {
        badgeContainer.classList.add("hidden");
        badgeContainer.innerHTML = "";
    }
    const input = document.getElementById("prescriptionFileInput");
    if (input) input.value = "";
}


// ============================================================
// TREATMENT COMPARISON (PHASE 4 - DYNAMIC MULTI-DISEASE)
// ============================================================

async function openTreatmentModal() {
    const modal = document.getElementById("treatmentModal");
    const input = document.getElementById("treatmentConditionInput");

    const defaultCond = lastActiveQuestion ? lastActiveQuestion.slice(0, 50) : "Asthma";
    if (input) input.value = defaultCond;

    modal.classList.remove("hidden");
    executeTreatmentCompare();
}

async function executeTreatmentCompare() {
    const input = document.getElementById("treatmentConditionInput");
    const bodyEl = document.getElementById("treatmentBody");

    const conditionText = (input && input.value.trim()) ? input.value.trim() : (lastActiveQuestion || "Asthma");
    bodyEl.innerHTML = `<p>Analyzing multi-source medical evidence for <strong>${conditionText}</strong>...</p>`;

    try {
        const resp = await fetch(TREATMENT_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                condition: conditionText,
                language: document.getElementById("language").value || "English"
            })
        });

        if (!resp.ok) throw new Error("Could not load treatment comparison.");
        const data = await resp.json();

        if (!data.options || data.options.length === 0) {
            bodyEl.innerHTML = `
                <p><strong>Condition:</strong> ${data.condition}</p>
                <p style="color:#d5e6f5; margin:14px 0;">${data.overview}</p>
                <div class="report-disclaimer-box">
                    <strong>CLINICAL NOTICE:</strong> ${data.disclaimer}
                </div>
            `;
            return;
        }

        bodyEl.innerHTML = `
            <p><strong>Condition Evaluated:</strong> ${data.condition}</p>
            <p style="font-size:13.5px; color:#afcddc; margin-bottom:16px;">${data.overview}</p>
            <div class="treatment-table-wrapper">
                <table class="treatment-table">
                    <thead>
                        <tr>
                            <th>Treatment Approach</th>
                            <th>How It Works (Mechanism)</th>
                            <th>Main Purpose</th>
                            <th>Clinical Benefits</th>
                            <th>Common Side Effects</th>
                            <th>Risks & Precautions</th>
                            <th>Suitability Considerations</th>
                            <th>Administration Method</th>
                            <th>Evidence References</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.options.map(opt => `
                            <tr>
                                <td><strong>${opt.treatment}</strong></td>
                                <td>${opt.how_it_works}</td>
                                <td>${opt.main_purpose}</td>
                                <td>${opt.benefits}</td>
                                <td>${opt.side_effects}</td>
                                <td>${opt.risks_precautions}</td>
                                <td>${opt.suitability}</td>
                                <td>${opt.administration}</td>
                                <td><em>${Array.isArray(opt.evidence_sources) ? opt.evidence_sources.join(", ") : opt.evidence_sources}</em></td>
                            </tr>
                        `).join("")}
                    </tbody>
                </table>
            </div>
            <p style="font-size:11.5px; color:#8ab3c9; margin-top:16px;">⚠️ ${data.disclaimer}</p>
        `;
    } catch (err) {
        bodyEl.innerHTML = `<p style="color:#ff8585;">Failed to load comparison: ${err.message}</p>`;
    }
}


// ============================================================
// SELECT ANY MEDICAL WORD OR SENTENCE TO EXPLAIN (PHASE 6)
// ============================================================

function setupSelectionExplanation() {
    const pill = document.getElementById("wordSelectionPill");

    const handleSelection = () => {
        const selection = window.getSelection();
        if (!selection || selection.isCollapsed) {
            if (pill && !pill.matches(":hover")) pill.classList.add("hidden");
            return;
        }

        const selectedStr = selection.toString().trim();
        if (selectedStr.length < 3 || selectedStr.length > 150) {
            if (pill) pill.classList.add("hidden");
            return;
        }

        // Verify selection is within an answer area
        const range = selection.getRangeAt(0);
        const container = range.commonAncestorContainer;
        const answerArea = (container.nodeType === 1 ? container : container.parentElement).closest(".selectable-answer-area, .answer");

        if (!answerArea) {
            if (pill) pill.classList.add("hidden");
            return;
        }

        currentSelectedText = selectedStr;
        currentSelectedContext = answerArea.innerText ? answerArea.innerText.slice(0, 300) : "";

        // Position floating pill above selection
        const rect = range.getBoundingClientRect();
        if (pill) {
            pill.style.top = `${window.scrollY + rect.top - 42}px`;
            pill.style.left = `${window.scrollX + rect.left + (rect.width / 2) - 60}px`;
            pill.classList.remove("hidden");
        }
    };

    document.addEventListener("mouseup", handleSelection);
    document.addEventListener("touchend", handleSelection);
}

async function explainCurrentSelection() {
    const pill = document.getElementById("wordSelectionPill");
    if (pill) pill.classList.add("hidden");

    if (!currentSelectedText) return;

    const modal = document.getElementById("wordExplanationModal");
    const titleEl = document.getElementById("selectionTermTitle");
    const defEl = document.getElementById("selectionDefinition");
    const impEl = document.getElementById("selectionImportance");
    const exEl = document.getElementById("selectionExample");

    titleEl.textContent = `Explaining "${currentSelectedText}"...`;
    defEl.textContent = "Loading plain-language explanation...";
    if (impEl) impEl.textContent = "";
    if (exEl) exEl.textContent = "";

    modal.classList.remove("hidden");

    try {
        const resp = await fetch(EXPLAIN_SELECTION_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                selected_text: currentSelectedText,
                context: currentSelectedContext,
                language: document.getElementById("language")?.value || "English"
            })
        });

        if (!resp.ok) throw new Error("Could not retrieve definition.");
        const data = await resp.json();

        titleEl.textContent = data.term;
        defEl.textContent = data.explanation;
        if (impEl) impEl.textContent = data.importance || "";
        if (exEl) exEl.textContent = data.example || "";

    } catch (err) {
        defEl.textContent = `Could not explain term: ${err.message}`;
    }
}

function closeWordExplanation() {
    const modal = document.getElementById("wordExplanationModal");
    if (modal) modal.classList.add("hidden");
    const pill = document.getElementById("wordSelectionPill");
    if (pill) pill.classList.add("hidden");
}


// ============================================================
// DOCTOR DISCUSSION SUMMARY MODAL
// ============================================================

async function openDoctorModal() {
    const modal = document.getElementById("doctorModal");
    const bodyEl = document.getElementById("doctorBody");

    const questionText = lastActiveQuestion || "General health consultation";
    modal.classList.remove("hidden");
    bodyEl.innerHTML = "<p>Preparing doctor discussion checklist...</p>";

    try {
        const resp = await fetch(DOCTOR_SUMMARY_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: questionText,
                answer: (lastActiveResponse && lastActiveResponse.answer) || "",
                sources: (lastActiveResponse && lastActiveResponse.sources) || []
            })
        });

        if (!resp.ok) throw new Error("Could not prepare doctor summary.");
        const data = await resp.json();

        bodyEl.innerHTML = `
            <div class="report-section">
                <h3>Consultation Summary</h3>
                <p>${data.summary_for_doctor}</p>
            </div>

            <div class="report-section">
                <h3>Key Questions to Ask Your Healthcare Provider</h3>
                <ol style="padding-left:20px; line-height:1.8;">
                    ${data.questions_to_ask.map(q => `<li><strong>${q}</strong></li>`).join("")}
                </ol>
            </div>

            <div class="report-section">
                <h3>Important Red-Flag Symptoms to Disclose</h3>
                <ul style="padding-left:20px; color:#ff9999;">
                    ${data.critical_symptoms_to_mention.map(s => `<li>${s}</li>`).join("")}
                </ul>
            </div>

            <div style="margin-top:18px; font-size:12px; color:#8bb2c6;">
                ℹ️ ${data.disclaimer}
            </div>
        `;
    } catch (err) {
        bodyEl.innerHTML = `<p style="color:#ff8585;">Failed to load summary: ${err.message}</p>`;
    }
}


// ============================================================
// AI-GENERATED STRUCTURED MEDICAL REPORT MODAL (PREVIEW & PRINT)
// ============================================================

async function openReportModal() {
    if (!lastActiveResponse) {
        showToast("Please ask a question first to generate a report.");
        return;
    }

    const modal = document.getElementById("reportModal");
    const titleEl = document.getElementById("modalReportTitle");
    const bodyEl = document.getElementById("reportBody");

    bodyEl.innerHTML = "<p>Generating structured patient education report...</p>";
    modal.classList.remove("hidden");

    try {
        const resp = await fetch(REPORT_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: lastActiveQuestion,
                answer: lastActiveResponse.answer,
                sources: lastActiveResponse.sources || [],
                confidence: lastActiveResponse.confidence || "Unknown",
                safety_level: lastActiveResponse.safety_level || "Normal",
                safety_message: lastActiveResponse.safety_message || "",
                language: document.getElementById("language").value || "English"
            })
        });

        if (!resp.ok) throw new Error("Could not generate report.");
        const report = await resp.json();

        titleEl.textContent = report.report_title;

        let termsHTML = "";
        if (report.medical_terms && report.medical_terms.length) {
            termsHTML = `
                <div class="report-section">
                    <h3>Medical Terminology</h3>
                    ${report.medical_terms.map(t => `<p><strong>${t.term}:</strong> ${t.definition} <em>(Why it matters: ${t.importance})</em></p>`).join("")}
                </div>
            `;
        }

        let factsHTML = "";
        if (report.key_facts && report.key_facts.length) {
            factsHTML = `
                <div class="report-section">
                    <h3>Key Evidence-Based Points</h3>
                    <ul>${report.key_facts.map(f => `<li>${f}</li>`).join("")}</ul>
                </div>
            `;
        }

        let sourcesHTML = "";
        if (report.sources && report.sources.length) {
            sourcesHTML = `
                <div class="report-section">
                    <h3>Medical References</h3>
                    <ul>${report.sources.map(s => `<li>${s.title} (${s.sources}) ${s.url ? `— <a href="${s.url}" target="_blank">${s.url}</a>` : ""}</li>`).join("")}</ul>
                </div>
            `;
        }

        bodyEl.innerHTML = `
            <div class="report-content-print">
                <div class="report-section">
                    <h3>Patient Inquired Question</h3>
                    <p style="font-size:15px; font-weight:600;">"${report.question}"</p>
                </div>

                <div class="report-section">
                    <h3>Plain-Language Explanation</h3>
                    <p>${formatMedicalAnswer(report.plain_explanation)}</p>
                </div>

                ${factsHTML}
                ${termsHTML}

                <div class="report-section">
                    <h3>Precautions & Safety Guidance</h3>
                    <ul>${report.precautions.map(p => `<li>${p}</li>`).join("")}</ul>
                </div>

                ${sourcesHTML}

                <div class="report-section" style="margin-top:24px; padding:14px; background:rgba(0,0,0,0.1); border-left:4px solid #08cbbd; font-size:12px; color:#a2c4d6;">
                    <strong>CLINICAL DISCLAIMER:</strong> ${report.disclaimer}
                </div>
            </div>
        `;
    } catch (err) {
        bodyEl.innerHTML = `<p style="color:#ff8585;">Failed to load report: ${err.message}</p>`;
    }
}


// ============================================================
// COPY ANSWER FUNCTIONALITY
// ============================================================

async function copyAnswer() {
    const answerEl = document.getElementById("answer");
    const copyBtn = document.getElementById("copyButton");

    if (!answerEl || !answerEl.textContent.trim()) {
        showToast("No answer text to copy.");
        return;
    }

    const textToCopy = answerEl.innerText || answerEl.textContent;

    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(textToCopy);
        } else {
            const textArea = document.createElement("textarea");
            textArea.value = textToCopy;
            textArea.style.position = "fixed";
            textArea.style.left = "-999999px";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            document.execCommand("copy");
            textArea.remove();
        }

        showToast("✓ Medical explanation copied to clipboard!");
        if (copyBtn) {
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = "✓ Copied!";
            copyBtn.style.borderColor = "#21e0d5";
            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
                copyBtn.style.borderColor = "";
            }, 2500);
        }
    } catch (err) {
        console.error("Failed to copy:", err);
        showToast("Unable to copy to clipboard.");
    }
}


// ============================================================
// MODAL CONTROLS
// ============================================================

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.add("hidden");
}

function closeModalOnBackdrop(event, modalId) {
    if (event.target && event.target.id === modalId) {
        closeModal(modalId);
    }
}


// ============================================================
// SPEECH SYNTHESIS & VOICE INPUT
// ============================================================

function toggleSpeech() {
    const speakBtn = document.getElementById("speakButton");
    const answerEl = document.getElementById("answer");

    if (!window.speechSynthesis) {
        showToast("Speech synthesis is not supported in this browser.");
        return;
    }

    if (window.speechSynthesis.speaking) {
        stopSpeech();
        return;
    }

    if (!answerEl || !answerEl.textContent.trim()) {
        showToast("No answer to read aloud.");
        return;
    }

    const cleanText = answerEl.innerText || answerEl.textContent;
    currentSpeechUtterance = new SpeechSynthesisUtterance(cleanText);

    const lang = document.getElementById("language").value || "English";
    if (lang === "Kannada") currentSpeechUtterance.lang = "kn-IN";
    else if (lang === "Hindi") currentSpeechUtterance.lang = "hi-IN";
    else currentSpeechUtterance.lang = "en-US";

    currentSpeechUtterance.onstart = () => {
        if (speakBtn) speakBtn.innerHTML = "⏹️ Stop";
    };

    currentSpeechUtterance.onend = () => {
        if (speakBtn) speakBtn.innerHTML = "🔊 Listen";
    };

    currentSpeechUtterance.onerror = () => {
        if (speakBtn) speakBtn.innerHTML = "🔊 Listen";
    };

    window.speechSynthesis.speak(currentSpeechUtterance);
}

function stopSpeech() {
    if (window.speechSynthesis && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
    }
    const speakBtn = document.getElementById("speakButton");
    if (speakBtn) speakBtn.innerHTML = "🔊 Listen";
}

function toggleVoiceInput() {
    const voiceBtn = document.getElementById("voiceInputBtn");
    const questionInput = document.getElementById("question");

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showToast("Voice input is not supported in your browser.");
        return;
    }

    if (isRecordingVoice) {
        if (speechRecognitionInstance) speechRecognitionInstance.stop();
        return;
    }

    speechRecognitionInstance = new SpeechRecognition();
    const lang = document.getElementById("language").value || "English";
    if (lang === "Kannada") speechRecognitionInstance.lang = "kn-IN";
    else if (lang === "Hindi") speechRecognitionInstance.lang = "hi-IN";
    else speechRecognitionInstance.lang = "en-US";

    speechRecognitionInstance.interimResults = false;
    speechRecognitionInstance.maxAlternatives = 1;

    speechRecognitionInstance.onstart = () => {
        isRecordingVoice = true;
        if (voiceBtn) voiceBtn.classList.add("recording");
        showToast("Listening... Speak your medical question.");
    };

    speechRecognitionInstance.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (questionInput) {
            questionInput.value = questionInput.value ? (questionInput.value + " " + transcript) : transcript;
            questionInput.focus();
        }
    };

    speechRecognitionInstance.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        showToast(`Voice error: ${event.error}`);
    };

    speechRecognitionInstance.onend = () => {
        isRecordingVoice = false;
        if (voiceBtn) voiceBtn.classList.remove("recording");
    };

    speechRecognitionInstance.start();
}

function setQuestion(text) {
    const q = document.getElementById("question");
    if (q) {
        q.value = text;
        q.focus();
        q.scrollIntoView({ behavior: "smooth", block: "center" });
    }
}


// ============================================================
// DOM INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Chats
    medicalChats = loadMedicalChats();
    renderMedicalChatHistory();
    syncConversationsWithBackend();

    // 2. Setup Selection-based Word Explanation (Phase 6)
    setupSelectionExplanation();

    // 3. Question counter & shortcuts
    const questionInput = document.getElementById("question");
    const questionCount = document.getElementById("questionCount");
    const askAnotherBtn = document.getElementById("askAnotherButton");

    if (questionInput && questionCount) {
        const updateCount = () => {
            questionCount.textContent = `${questionInput.value.length}/600`;
        };
        questionInput.addEventListener("input", updateCount);
        updateCount();

        questionInput.addEventListener("keydown", (event) => {
            if (event.ctrlKey && event.key === "Enter") {
                event.preventDefault();
                askMediExplain();
            }
        });
    }

    if (askAnotherBtn) {
        askAnotherBtn.addEventListener("click", () => {
            const q = document.getElementById("question");
            if (q) {
                q.value = "";
                q.focus();
                q.scrollIntoView({ behavior: "smooth", block: "center" });
            }
        });
    }

    // 4. Keyboard shortcuts (ESC closes modals)
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeModal("reportModal");
            closeModal("treatmentModal");
            closeModal("doctorModal");
            closeModal("pdfReportModal");
            closeModal("prescriptionModal");
            closeWordExplanation();
            toggleSidebar(false);
        }
    });
});
