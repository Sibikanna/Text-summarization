from flask import Flask, request, render_template
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

app = Flask(__name__)

# Load the Pegasus model and tokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["text"]

        # Encode the input text
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate the summary
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
            )

            # Decode the summary
            summary = tokenizer.decode(outputs[0])

            return render_template("index.html", input_text=input_text, summary=summary)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")