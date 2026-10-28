// ============================================================
// MEDIEXPLAIN AI - PROFESSIONAL FRONTEND CONTROLLER
// ============================================================

const BASE_API_URL = "http://127.0.0.1:8000/api";
const CHAT_API_URL = `${BASE_API_URL}/chat`;
const UPLOAD_DOC_URL = `${BASE_API_URL}/upload-doc`;
const REPORT_API_URL = `${BASE_API_URL}/generate-report`;
const TREATMENT_API_URL = `${BASE_API_URL}/treatment-compare`;
const DOCTOR_SUMMARY_URL = `${BASE_API_URL}/doctor-summary`;

// Global State
let isRequestInProgress = false;
let activeDocId = null;
let activeDocName = null;
let attachedImageFile = null;
let currentSpeechUtterance = null;
let speechRecognitionInstance = null;
let isRecordingVoice = false;

// Current Active Interaction Data for Reports & Modals
let lastActiveResponse = null;
let lastActiveQuestion = "";

// Chat History Storage
const CHAT_HISTORY_KEY = "mediexplain_chat_history_v2";
let activeChatId = null;
let medicalChats = loadMedicalChats();


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
// MAIN QUESTION SUBMISSION (ASK MEDIEXPLAIN)
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

    const question = questionInput.value.trim();
    const language = languageInput.value || "English";

    // 1. Validation
    if (!question) {
        showToast("Please enter a medical question.");
        questionInput.focus();
        return;
    }

    clearMedicalError();
    isRequestInProgress = true;

    // Stop ongoing speech
    stopSpeech();

    // UI state: loading
    if (responseSection) responseSection.classList.add("hidden");
    if (loading) loading.classList.remove("hidden");
    if (askButton) {
        askButton.disabled = true;
        askButton.innerHTML = "<span>Analyzing...</span> <span>⏳</span>";
    }

    // Scroll loading into view
    loading.scrollIntoView({ behavior: "smooth", block: "center" });

    // Prepare multi-turn history from active chat (last 3 turns max)
    const historyPayload = [];
    if (activeChatId) {
        const currentChat = medicalChats.find(c => c.id === activeChatId);
        if (currentChat && currentChat.messages && currentChat.messages.length) {
            const recent = currentChat.messages.slice(-3);
            recent.forEach(m => {
                if (m.question) historyPayload.push({ role: "user", content: m.question });
                if (m.answer) historyPayload.push({ role: "assistant", content: m.answer.slice(0, 300) });
            });
        }
    }

    try {
        const response = await fetch(CHAT_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: question,
                language: language,
                history: historyPayload.length ? historyPayload : null,
                doc_id: activeDocId || null
            })
        });

        // 2. Handle HTTP Errors
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
            return;
        }

        const data = await response.json();
        lastActiveResponse = data;
        lastActiveQuestion = question;

        // 3. Save to Active Chat Conversation
        if (!activeChatId) {
            createMedicalChat();
        }

        const currentChat = medicalChats.find(c => c.id === activeChatId);
        if (currentChat) {
            currentChat.title = currentChat.messages.length === 0
                ? (question.slice(0, 42) + (question.length > 42 ? "..." : ""))
                : currentChat.title;

            currentChat.messages.push({
                question: question,
                answer: data.answer || "No explanation received.",
                confidence: data.confidence || "Unknown",
                safetyLevel: data.safety_level || "Normal",
                safetyMessage: data.safety_message || "",
                sources: data.sources || [],
                evidence_explorer: data.evidence_explorer || [],
                terminology: data.terminology || [],
                language: language,
                timestamp: new Date().toISOString()
            });

            currentChat.updatedAt = new Date().toISOString();
            saveMedicalChats();
            renderMedicalChatHistory();
        }

        // 4. Render Explanation & Metrics
        answer.innerHTML = formatMedicalAnswer(data.answer || "No explanation received.");

        confidence.textContent = data.confidence || "Unknown";
        confidence.className = `confidence-${String(data.confidence || "unknown").toLowerCase()}`;

        safetyLevel.textContent = data.safety_level || "Normal";
        safetyLevel.className = `safety-${String(data.safety_level || "normal").toLowerCase()}`;

        safetyMessage.textContent = data.safety_message || "This information is for educational purposes only.";

        // Special Emergency UI Styling
        const safetyCard = document.querySelector(".safety-card");
        if (safetyCard) {
            safetyCard.classList.remove("safety-emergency", "safety-warning", "safety-normal");
            const level = String(data.safety_level || "Normal").toLowerCase();
            if (level === "emergency") safetyCard.classList.add("safety-emergency");
            else if (level === "warning") safetyCard.classList.add("safety-warning");
            else safetyCard.classList.add("safety-normal");
        }

        // 5. Render Medical Terminology Simplifier (Phase 3D)
        renderTerminology(data.terminology || []);

        // 6. Render Evidence Explorer (Phase 5)
        renderEvidenceExplorer(data.evidence_explorer || []);

        // 7. Render Citations & Sources
        renderSources(sources, data.sources || []);

        // 8. Reveal Response Section
        responseSection.classList.remove("hidden");
        responseSection.scrollIntoView({ behavior: "smooth", block: "start" });

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
    } finally {
        // Reset loading states strictly
        if (loading) loading.classList.add("hidden");
        if (askButton) {
            askButton.disabled = false;
            askButton.innerHTML = "<span>Ask MediExplain</span> <span>→</span>";
        }
        isRequestInProgress = false;
    }
}


// ============================================================
// COPY ANSWER FUNCTIONALITY (STRICTLY ONE VISIBLE BUTTON)
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
            // Fallback for older browsers
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
// FORMAT MEDICAL ANSWER
// ============================================================

function formatMedicalAnswer(text) {
    if (!text) return "No answer received.";

    // HTML escape
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
// RENDER MEDICAL SOURCES (CITATIONS)
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

        // Claims mapping display if present
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


// ============================================================
// RENDER EVIDENCE EXPLORER (PHASE 5)
// ============================================================

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

        item.innerHTML = `
            <div class="evidence-item-top">
                <span class="evidence-title">Passage ${idx + 1}: ${ev.title}</span>
                <span class="evidence-score-badge">Relevance: ${scorePercent}%</span>
            </div>
            <div class="evidence-excerpt">"${ev.excerpt || "No excerpt extracted."}"</div>
            ${ev.url ? `<a href="${ev.url}" target="_blank" rel="noopener noreferrer" class="evidence-source-link">Source link: ${ev.title} →</a>` : ""}
        `;

        container.appendChild(item);
    });
}


// ============================================================
// RENDER TERMINOLOGY SIMPLIFIER (PHASE 3D)
// ============================================================

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
// PDF DOCUMENT ATTACHMENT (PHASE 7)
// ============================================================

async function handlePdfUpload(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showToast("Only PDF documents are supported.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showToast("PDF file exceeds the 10 MB limit.");
        return;
    }

    const badgeContainer = document.getElementById("attachedDocContainer");
    badgeContainer.classList.remove("hidden");
    badgeContainer.innerHTML = `<span class="attached-doc-badge">⏳ Uploading & parsing ${file.name}...</span>`;

    try {
        const base64Data = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const res = reader.result;
                const base64 = typeof res === "string" && res.includes(",") ? res.split(",")[1] : res;
                resolve(base64);
            };
            reader.onerror = () => reject(new Error("Failed to read file"));
            reader.readAsDataURL(file);
        });

        const resp = await fetch(UPLOAD_DOC_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                filename: file.name,
                file_base64: base64Data
            })
        });
            body: formData
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail ? (err.detail.message || JSON.stringify(err.detail)) : `Upload failed (${resp.status})`);
        }

        const data = await resp.json();
        activeDocId = data.doc_id;
        activeDocName = data.filename;

        badgeContainer.innerHTML = `
            <div class="attached-doc-badge">
                <span>📄 ${data.filename} (${data.pages} pages)</span>
                <button type="button" class="attached-remove-btn" onclick="removeAttachedDoc()" title="Remove document">×</button>
            </div>
        `;
        showToast(`Document parsed! You can now ask questions about ${data.filename}.`);
    }
    catch (err) {
        console.error("PDF upload error:", err);
        showToast(`PDF upload failed: ${err.message}`);
        badgeContainer.classList.add("hidden");
        badgeContainer.innerHTML = "";
    }
}

function removeAttachedDoc() {
    activeDocId = null;
    activeDocName = null;
    const badgeContainer = document.getElementById("attachedDocContainer");
    if (badgeContainer) {
        badgeContainer.classList.add("hidden");
        badgeContainer.innerHTML = "";
    }
    const input = document.getElementById("pdfFileInput");
    if (input) input.value = "";
    showToast("Attached document removed.");
}


// ============================================================
// MEDICAL IMAGE ATTACHMENT (PHASE 8)
// ============================================================

function handleImageAttach(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    if (file.size > 5 * 1024 * 1024) {
        showToast("Image exceeds 5 MB limit.");
        return;
    }

    attachedImageFile = file;
    const container = document.getElementById("attachedImageContainer");
    const reader = new FileReader();

    reader.onload = function(e) {
        container.classList.remove("hidden");
        container.innerHTML = `
            <div style="position:relative; display:inline-block;">
                <img src="${e.target.result}" class="attached-image-preview" alt="Attached medical reference">
                <button type="button" class="attached-remove-btn" style="position:absolute; top:4px; right:4px; background:rgba(0,0,0,0.7); border-radius:50%;" onclick="removeAttachedImage()">×</button>
            </div>
            <div class="attached-image-notice">🖼️ Educational reference attached. MediExplain AI does not diagnose from images.</div>
        `;
        showToast("Image attached for medical context.");
    };

    reader.readAsDataURL(file);
}

function removeAttachedImage() {
    attachedImageFile = null;
    const container = document.getElementById("attachedImageContainer");
    if (container) {
        container.classList.add("hidden");
        container.innerHTML = "";
    }
    const input = document.getElementById("imageFileInput");
    if (input) input.value = "";
    showToast("Attached image removed.");
}


// ============================================================
// AI-GENERATED STRUCTURED MEDICAL REPORT (PHASE 9)
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
// TREATMENT COMPARISON MODAL (PHASE 10)
// ============================================================

async function openTreatmentModal() {
    const modal = document.getElementById("treatmentModal");
    const bodyEl = document.getElementById("treatmentBody");

    const conditionText = lastActiveQuestion || "diabetes";
    modal.classList.remove("hidden");
    bodyEl.innerHTML = "<p>Analyzing evidence-based treatment options...</p>";

    try {
        const resp = await fetch(TREATMENT_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ condition: conditionText })
        });

        if (!resp.ok) throw new Error("Could not load treatment comparison.");
        const data = await resp.json();

        bodyEl.innerHTML = `
            <p><strong>Condition:</strong> ${data.condition}</p>
            <p style="font-size:13.5px; color:#afcddc; margin-bottom:16px;">${data.overview}</p>
            <div class="treatment-table-wrapper">
                <table class="treatment-table">
                    <thead>
                        <tr>
                            <th>Treatment Approach</th>
                            <th>Category</th>
                            <th>Clinical Benefits</th>
                            <th>Limitations</th>
                            <th>Known Risks / Side Effects</th>
                            <th>Reference</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.options.map(opt => `
                            <tr>
                                <td><strong>${opt.treatment}</strong></td>
                                <td>${opt.category}</td>
                                <td>${opt.benefits}</td>
                                <td>${opt.limitations}</td>
                                <td>${opt.risks}</td>
                                <td><em>${opt.evidence_source}</em></td>
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
// DOCTOR DISCUSSION SUMMARY MODAL (PHASE 10)
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
// SPEECH SYNTHESIS (READ ALOUD)
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


// ============================================================
// SPEECH RECOGNITION (VOICE INPUT)
// ============================================================

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


// ============================================================
// CHAT HISTORY STORAGE & MANAGEMENT
// ============================================================

function loadMedicalChats() {
    try {
        const saved = localStorage.getItem(CHAT_HISTORY_KEY);
        return saved ? JSON.parse(saved) : [];
    } catch (e) {
        console.error("Unable to load chat history:", e);
        return [];
    }
}

function saveMedicalChats() {
    try {
        localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(medicalChats));
    } catch (e) {
        console.error("Unable to save chat history:", e);
    }
}

function createMedicalChat() {
    const chat = {
        id: "chat_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7),
        title: "New Medical Chat",
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    };

    medicalChats.unshift(chat);
    activeChatId = chat.id;

    saveMedicalChats();
    renderMedicalChatHistory();
    return chat;
}

function renderMedicalChatHistory() {
    const list = document.getElementById("chatHistoryList");
    if (!list) return;

    list.innerHTML = "";

    if (medicalChats.length === 0) {
        list.innerHTML = "<p class='chat-history-empty' style='color:#759eb5; font-size:13px; padding:12px 14px;'>No saved conversations yet.</p>";
        return;
    }

    medicalChats.forEach(chat => {
        const item = document.createElement("div");
        item.className = "chat-history-item";
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
        renameBtn.onclick = (e) => { e.stopPropagation(); renameMedicalChat(chat.id); };

        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.innerHTML = "×";
        delBtn.title = "Delete conversation";
        delBtn.onclick = (e) => { e.stopPropagation(); deleteMedicalChat(chat.id); };

        actions.appendChild(renameBtn);
        actions.appendChild(delBtn);
        item.appendChild(openBtn);
        item.appendChild(actions);
        list.appendChild(item);
    });
}

function openMedicalChat(chatId) {
    const chat = medicalChats.find(c => c.id === chatId);
    if (!chat) return;

    activeChatId = chat.id;
    clearMedicalChatDisplay();

    const welcome = document.getElementById("chatWelcome");
    if (welcome) welcome.style.display = chat.messages.length ? "none" : "";

    const questionInput = document.getElementById("question");
    const answer = document.getElementById("answer");
    const sources = document.getElementById("sources");
    const response = document.getElementById("responseSection");
    const confidence = document.getElementById("confidence");
    const safetyLevel = document.getElementById("safetyLevel");
    const safetyMessage = document.getElementById("safetyMessage");

    if (chat.messages.length) {
        const lastMsg = chat.messages[chat.messages.length - 1];
        lastActiveResponse = lastMsg;
        lastActiveQuestion = lastMsg.question;

        if (questionInput) questionInput.value = lastMsg.question || "";
        if (answer) answer.innerHTML = formatMedicalAnswer(lastMsg.answer);
        if (confidence) confidence.textContent = lastMsg.confidence || "Unknown";
        if (safetyLevel) safetyLevel.textContent = lastMsg.safetyLevel || "Normal";
        if (safetyMessage) safetyMessage.textContent = lastMsg.safetyMessage || "";
        if (sources) renderSources(sources, lastMsg.sources || []);
        renderEvidenceExplorer(lastMsg.evidence_explorer || []);
        renderTerminology(lastMsg.terminology || []);
        if (response) response.classList.remove("hidden");
    }

    renderMedicalChatHistory();
}

function renameMedicalChat(chatId) {
    const chat = medicalChats.find(c => c.id === chatId);
    if (!chat) return;

    const newTitle = prompt("Enter a title for this medical chat:", chat.title);
    if (newTitle === null) return;

    const cleaned = newTitle.trim();
    if (!cleaned) {
        showToast("Title cannot be empty.");
        return;
    }

    chat.title = cleaned.slice(0, 60);
    chat.updatedAt = new Date().toISOString();
    saveMedicalChats();
    renderMedicalChatHistory();
}

function deleteMedicalChat(chatId) {
    const chat = medicalChats.find(c => c.id === chatId);
    if (!chat) return;

    if (!confirm(`Delete conversation "${chat.title}"?`)) return;

    medicalChats = medicalChats.filter(c => c.id !== chatId);
    if (activeChatId === chatId) {
        activeChatId = null;
        clearMedicalChatDisplay();
    }

    saveMedicalChats();
    renderMedicalChatHistory();
    showToast("Conversation deleted.");
}

function startNewMedicalChat() {
    activeChatId = null;
    clearMedicalChatDisplay();

    const welcome = document.getElementById("chatWelcome");
    if (welcome) welcome.style.display = "";

    const questionInput = document.getElementById("question");
    if (questionInput) questionInput.focus();

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
    removeAttachedImage();
    clearMedicalError();
    stopSpeech();
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
    const newChatBtn = document.getElementById("newChatButton");
    const sidebarToggle = document.getElementById("sidebarToggle");
    const questionInput = document.getElementById("question");
    const questionCount = document.getElementById("questionCount");
    const askAnotherBtn = document.getElementById("askAnotherButton");

    if (newChatBtn) {
        newChatBtn.addEventListener("click", startNewMedicalChat);
    }

    if (sidebarToggle) {
        sidebarToggle.addEventListener("click", () => {
            const sidebar = document.getElementById("chatSidebar");
            if (sidebar) sidebar.classList.toggle("sidebar-open");
        });
    }

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

    // Load initial chat history
    renderMedicalChatHistory();
});
