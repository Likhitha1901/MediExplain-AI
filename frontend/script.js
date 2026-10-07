// ============================================================
// MEDIEXPLAIN AI - FRONTEND SCRIPT
// ============================================================

const API_URL = "http://127.0.0.1:8000/api/chat";


// ============================================================
// ASK MEDIEXPLAIN
// ============================================================

async function askMediExplain() {

    const questionInput = document.getElementById("question");
    const languageInput = document.getElementById("language");

    const answer = document.getElementById("answer");
    const sources = document.getElementById("sources");
    const confidence = document.getElementById("confidence");
    const safetyLevel = document.getElementById("safetyLevel");
    const safetyMessage = document.getElementById("safetyMessage");

    const responseSection =
        document.getElementById("responseSection");

    const loading =
        document.getElementById("loading");

    const askButton =
        document.getElementById("askButton");


    const question = questionInput.value.trim();
    const language = languageInput.value;


    // ========================================================
    // CHECK QUESTION
    // ========================================================

    if (!question) {

        alert("Please enter a medical question.");

        questionInput.focus();

        return;
    }


    // ========================================================
    // SHOW LOADING
    // ========================================================

    responseSection.classList.add("hidden");

    loading.classList.remove("hidden");

    askButton.disabled = true;

    askButton.textContent = "Analyzing...";


    try {

        // ====================================================
        // SEND REQUEST TO FASTAPI
        // ====================================================

        const response = await fetch(
            API_URL,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    message: question,
                    language: language
                })
            }
        );


        // ====================================================
        // CHECK SERVER RESPONSE
        // ====================================================

        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );
        }


        const data = await response.json();


        // ====================================================
        // DISPLAY ANSWER
        // ====================================================

        answer.innerHTML = formatMedicalAnswer(
            data.answer || "No answer received."
        );


        // ====================================================
        // DISPLAY CONFIDENCE
        // ====================================================

        confidence.textContent =
            data.confidence || "Unknown";

        confidence.className =
            `confidence-${String(
                data.confidence || "unknown"
            ).toLowerCase()}`;


        // ====================================================
        // DISPLAY SAFETY LEVEL
        // ====================================================

        safetyLevel.textContent =
            data.safety_level || "Unknown";

        safetyLevel.className =
            `safety-${String(
                data.safety_level || "unknown"
            ).toLowerCase()}`;


        // ====================================================
        // DISPLAY SAFETY MESSAGE
        // ====================================================

        safetyMessage.textContent =
            data.safety_message ||
            "No additional safety information.";


        // ====================================================
        // SPECIAL EMERGENCY UI
        // ====================================================

        const safetyCard =
            document.querySelector(".safety-card");

        if (safetyCard) {

            safetyCard.classList.remove(
                "safety-emergency",
                "safety-warning",
                "safety-normal"
            );

            const level =
                String(
                    data.safety_level || "Normal"
                ).toLowerCase();

            if (level === "emergency") {

                safetyCard.classList.add(
                    "safety-emergency"
                );

            } else if (
                level === "warning" ||
                level === "caution"
            ) {

                safetyCard.classList.add(
                    "safety-warning"
                );

            } else {

                safetyCard.classList.add(
                    "safety-normal"
                );
            }
        }


        // ====================================================
        // DISPLAY SOURCES
        // ====================================================

        renderSources(
            sources,
            data.sources || []
        );


        // ====================================================
        // ADD COPY BUTTON
        // ====================================================

        addCopyButton(
            answer,
            data.answer || ""
        );


        // ====================================================
        // ADD ASK AGAIN BUTTON
        // ====================================================

        addAskAgainButton();


        // ====================================================
        // SHOW RESPONSE
        // ====================================================

        responseSection.classList.remove("hidden");

        responseSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });


    } catch (error) {

        console.error(
            "MediExplain API Error:",
            error
        );


        alert(
            "Unable to connect to MediExplain AI.\n\n" +
            "Please make sure the FastAPI server is running."
        );

    } finally {

        // ====================================================
        // RESET LOADING
        // ====================================================

        loading.classList.add("hidden");

        askButton.disabled = false;

        askButton.textContent =
            "Ask MediExplain →";
    }
}


// ============================================================
// FORMAT MEDICAL ANSWER
// ============================================================

function formatMedicalAnswer(text) {

    if (!text) {
        return "No answer received.";
    }


    // Escape HTML first for safety
    let safeText = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");


    // Bold markdown
    safeText = safeText.replace(
        /\*\*(.*?)\*\*/g,
        "<strong>$1</strong>"
    );


    // Convert line breaks
    safeText = safeText.replace(
        /\n/g,
        "<br>"
    );


    // Convert simple bullet points
    safeText = safeText.replace(
        /(^|<br>)\s*[-•]\s+/g,
        "$1<span class=\"answer-bullet\">•</span> "
    );


    return safeText;
}


// ============================================================
// RENDER SOURCES
// ============================================================

function renderSources(container, sourceList) {

    container.innerHTML = "";


    if (
        !sourceList ||
        sourceList.length === 0
    ) {

        container.innerHTML = `
            <div class="no-sources">
                <span>📚</span>
                <p>No medical sources were returned.</p>
            </div>
        `;

        return;
    }


    sourceList.forEach(source => {

        const sourceItem =
            document.createElement("div");

        sourceItem.className =
            "source-item";


        // ----------------------------------------------------
        // ICON
        // ----------------------------------------------------

        const icon =
            document.createElement("div");

        icon.className =
            "source-icon";

        icon.textContent = "📚";


        // ----------------------------------------------------
        // CONTENT
        // ----------------------------------------------------

        const content =
            document.createElement("div");

        content.className =
            "source-content";


        // ----------------------------------------------------
        // TITLE
        // ----------------------------------------------------

        const title =
            document.createElement("div");

        title.className =
            "source-title";

        title.textContent =
            source.title ||
            "Medical Information";


        // ----------------------------------------------------
        // SOURCE + CATEGORY
        // ----------------------------------------------------

        const meta =
            document.createElement("div");

        meta.className =
            "source-meta";

        const sourceName =
            source.sources ||
            "Unknown source";

        const category =
            source.category ||
            "Medical information";


        meta.textContent =
            `${sourceName} • ${category}`;


        content.appendChild(title);

        content.appendChild(meta);
        // ----------------------------------------------------
// SOURCE LINK
// ----------------------------------------------------

        if (source.url) {

            const sourceLink =
                document.createElement("a");

            sourceLink.className =
                "source-link";

            sourceLink.textContent =
                "View source ↗";

            sourceLink.href =
                source.url;

            sourceLink.target =
                "_blank";

            sourceLink.rel =
                "noopener noreferrer";

            content.appendChild(sourceLink);
        }

        // ----------------------------------------------------
        // SOURCE LINK
        // ----------------------------------------------------

        if (source.url) {

            const link =
                document.createElement("a");

            link.className =
                "source-link";

            link.href =
                source.url;

            link.target =
                "_blank";

            link.rel =
                "noopener noreferrer";

            link.textContent =
                "View source →";


            content.appendChild(link);
        }


        sourceItem.appendChild(icon);

        sourceItem.appendChild(content);

        container.appendChild(sourceItem);

    });
}


// ============================================================
// COPY ANSWER BUTTON
// ============================================================

function addCopyButton(answerElement, originalText) {

    const answerCard =
        answerElement.closest(".answer-card");

    if (!answerCard) {
        return;
    }


    // Remove old button
    const oldButton =
        answerCard.querySelector(".copy-answer-button");

    if (oldButton) {
        oldButton.remove();
    }


    const button =
        document.createElement("button");

    button.className =
        "copy-answer-button";

    button.type =
        "button";

    button.textContent =
        "📋 Copy answer";


    button.onclick = async function () {

        try {

            await navigator.clipboard.writeText(
                originalText
            );

            button.textContent =
                "✓ Copied";

            setTimeout(() => {

                button.textContent =
                    "📋 Copy answer";

            }, 1800);

        } catch (error) {

            console.error(
                "Copy failed:",
                error
            );

            button.textContent =
                "Copy failed";
        }
    };


    answerCard.appendChild(button);
}


// ============================================================
// ASK ANOTHER QUESTION BUTTON
// ============================================================

function addAskAgainButton() {

    const responseSection =
        document.getElementById("responseSection");

    if (!responseSection) {
        return;
    }


    const oldButton =
        responseSection.querySelector(
            ".ask-again-button"
        );

    if (oldButton) {
        oldButton.remove();
    }


    const button =
        document.createElement("button");

    button.className =
        "ask-again-button";

    button.type =
        "button";

    button.textContent =
        "＋ Ask another question";


    button.onclick = function () {

        const question =
            document.getElementById("question");

        question.value = "";

        question.focus();

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    };


    responseSection.appendChild(button);
}


// ============================================================
// EXAMPLE QUESTION
// ============================================================

function setQuestion(text) {

    const questionBox =
        document.getElementById("question");

    questionBox.value = text;

    questionBox.focus();

    questionBox.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


// ============================================================
// ENTER KEY SUPPORT
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        const question =
            document.getElementById("question");


        if (question) {

            question.addEventListener(
                "keydown",
                function (event) {

                    // Ctrl + Enter = Ask
                    if (
                        event.ctrlKey &&
                        event.key === "Enter"
                    ) {

                        event.preventDefault();

                        askMediExplain();
                    }
                }
            );
        }
    }
);