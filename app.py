from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import rag_pipeline

# Store chat history
conversation = []

app = Flask(__name__)

# Enable CORS
CORS(app)


@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        print("Received Request:", data)

        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "answer": "Please enter a question."
            })

        # Save user message
        conversation.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Generate answer using memory
        answer = rag_pipeline(
            question,
            conversation
        )

        # Save bot reply
        conversation.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "answer": f"Backend Error: {str(e)}"
        }), 500


@app.route("/")
def home():

    return jsonify({
        "message": "Story Book Assistant Backend Running",
        "messages_in_memory": len(conversation)
    })


@app.route("/clear", methods=["POST"])
def clear_chat():

    conversation.clear()

    return jsonify({
        "message": "Conversation cleared"
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )