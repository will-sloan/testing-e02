import json

from e02_helpers.load_conversation_files import load_conversation_files
from e02_helpers.split_by_conversation import split_by_conversation
from e02_helpers.create_examples import create_examples
from e02_helpers.load_qwen_in_4bit import load_qwen_in_4bit
from e02_helpers.test_model import test_model
from e02_helpers.freeze_original_weights import freeze_original_weights
from e02_helpers.attach_trainable_adapter import attach_trainable_adapter
from e02_helpers.trainable_parameters import trainable_parameters
from e02_helpers.create_optimizer import create_optimizer
from e02_helpers.prepare_training_input import prepare_training_input
from e02_helpers.compare_results import compare_results
from e02_helpers.save_adapter import save_adapter

MAX_SAMPLES = 8  # Set to None to use all eligible samples.
SKIP_FIRST_CLINICIAN_MESSAGES = 2

def train_clinician_responses():
    if MAX_SAMPLES is None:
        max_train_examples = None
        max_test_examples = None
    else:
        if not isinstance(MAX_SAMPLES, int) or isinstance(MAX_SAMPLES, bool) or MAX_SAMPLES < 2:
            raise ValueError("MAX_SAMPLES must be None or a whole number of at least 2.")
        max_train_examples = max(1, int(MAX_SAMPLES * 0.8))
        max_test_examples = MAX_SAMPLES - max_train_examples

    # Step 1: Load JSON conversations or JSONL with one conversation per line.
    conversations = load_conversation_files(
        "real_conversations", max_conversations=MAX_SAMPLES,
    )


    # Keep whole conversations together so test examples are unseen in training.
    # If one participant has multiple conversations, keep those together too.
    train_conversations, test_conversations = split_by_conversation(conversations)

    # Each example is: conversation so far -> reviewed clinician reply.
    # The history must exclude that reply and every message after it.
    # Map clinician to assistant and user to user; no action labels are needed.
    train_examples = create_examples(
        train_conversations, max_examples=max_train_examples,
        skip_first_clinician_messages=SKIP_FIRST_CLINICIAN_MESSAGES,
    )
    test_examples = create_examples(
        test_conversations, max_examples=max_test_examples,
        skip_first_clinician_messages=SKIP_FIRST_CLINICIAN_MESSAGES,
    )

    total_samples = 0
    for conversation in conversations:
        messages = conversation["messages"] if isinstance(conversation, dict) else conversation
        clinician_messages = 0
        for index, message in enumerate(messages):
            fields = {key.lower(): value for key, value in message.items()}
            role = fields.get("speaker", fields.get("role", "")).strip().lower()
            text = fields.get("text", fields.get("content", ""))
            if role in {"clinician", "assistant"}:
                clinician_messages += 1
                if (index > 0 and clinician_messages > SKIP_FIRST_CLINICIAN_MESSAGES
                        and "[redacted]" not in text.casefold()):
                    total_samples += 1

    print(f"Training on {len(train_examples)}/{total_samples} eligible samples in loaded conversations", flush=True)
    print(f"Loaded {len(conversations)} conversations", flush=True)
    print(f"Skipping the first {SKIP_FIRST_CLINICIAN_MESSAGES} clinician messages per conversation as targets", flush=True)
    print(f"Using {len(train_examples)} training examples and {len(test_examples)} test examples", flush=True)
    for split, examples in [("Training", train_examples), ("Testing", test_examples)]:
        for index, (history, reply) in enumerate(examples, start=1):
            print(f"\n{split} example {index}:", flush=True)
            print(json.dumps({"input_messages": history, "target_reply": reply}, indent=2, ensure_ascii=False), flush=True)

    # Step 2: Load the project's Qwen model and its matching tokenizer.
    # The tokenizer converts messages into the token IDs the model processes.
    qwen, tokenizer = load_qwen_in_4bit("Qwen/Qwen3-4B-Instruct-2507")

    # Step 3: Test Qwen before training.
    # Generate from history only; use the clinician reply afterward for scoring.
    print("Testing the model before training...", flush=True)
    before_results = test_model(qwen, tokenizer, test_examples)

    # Step 4: Set up QLoRA fine-tuning.
    # Qwen stays frozen; the adapter learns adjustments inside Qwen.
    freeze_original_weights(qwen)
    model = attach_trainable_adapter(qwen)
    optimizer = create_optimizer(trainable_parameters(model))
    model.train()
    model.peft_config["default"].revision = "cdbee75f17c01a7cc42f958dc650907174af0554"

    # One pass through the training examples, for this simple outline.
    total_steps = len(train_examples)
    print(f"Starting training: {total_steps} steps", flush=True)
    for step, (conversation_so_far, clinician_reply) in enumerate(train_examples, start=1):
        # Use Qwen's chat template to turn history + reply into tokens.
        # Score only the reply and its end marker, not the history.
        # Causal masking prevents each prediction from seeing future tokens.
        batch = prepare_training_input(
            tokenizer,
            conversation_so_far,
            clinician_reply,
            train_on_reply_only=True,
        )

        device = model.get_input_embeddings().weight.device
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()          # Clear gradients from the previous example.
        loss = model(**batch).loss      # Measure errors predicting the reply.
        loss.backward()                # Calculate how parameters should change.
        optimizer.step()               # Update only the adapter's parameters.
        print(
            f"Training step {step}/{total_steps} ({step / total_steps:.0%}) "
            f"| Loss: {loss.item():.4f}",
            flush=True,
        )

    # Step 5: Test again using exactly the same held-out examples and settings.
    # test_model uses evaluation mode and disables gradient calculations.
    print("Training complete. Testing the adapted model...", flush=True)
    after_results = test_model(model, tokenizer, test_examples)
    compare_results(before_results, after_results)
    # Review response quality; different wording alone is not necessarily worse.

    # Step 6: Save the adapter for use together with the same Qwen base model.
    save_adapter(model, "saved_adapter")
    print("Adapter saved to saved_adapter", flush=True)


if __name__ == "__main__":
    train_clinician_responses()
