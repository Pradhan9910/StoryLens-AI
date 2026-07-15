const sendBtn = document.getElementById("send-btn");
const questionInput = document.getElementById("question");
const chatBox = document.getElementById("chat-box");

async function sendMessage() {

    let question = questionInput.value.trim();

    if (question === "") {
        return;
    }

    sendBtn.disabled = true;

    chatBox.innerHTML += `
        <div class="user-message">
            <b>You:</b> ${question}
        </div>
    `;

    questionInput.value = "";

    chatBox.innerHTML += `
        <div class="bot-message" id="thinking">
            <b>Story Bot:</b> Thinking...
        </div>
    `;

    chatBox.scrollTop = chatBox.scrollHeight;

    try {

        const response = await fetch(
            "http://127.0.0.1:5000/ask",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    question: question
                })
            }
        );

        const data = await response.json();

        document.getElementById("thinking").remove();

        chatBox.innerHTML += `
            <div class="bot-message">
                <b>Story Bot:</b> ${marked.parse(data.answer)}
            </div>
        `;

    } catch (error) {

        document.getElementById("thinking").remove();

        chatBox.innerHTML += `
            <div class="bot-message">
                <b>Story Bot:</b> Error connecting to backend.
            </div>
        `;
    }

    sendBtn.disabled = false;

    chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.addEventListener(
    "click",
    sendMessage
);

questionInput.addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {
            sendMessage();
        }
    }
);