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

def fine_tuning_pseudocode():
    # Step 1: Load JSON conversations or JSONL with one conversation per line.
    # conversations = load_conversation_files("C:/Users/sam3sim/Document/TestingData")
    conversations = load_conversation_files("sample_conversations")


    # Keep whole conversations together so test examples are unseen in training.
    # If one participant has multiple conversations, keep those together too.
    train_conversations, test_conversations = split_by_conversation(conversations)

    # Each example is: conversation so far -> reviewed clinician reply.
    # The history must exclude that reply and every message after it.
    # Map clinician to assistant and user to user; no action labels are needed.
    train_examples = create_examples(train_conversations)
    test_examples = create_examples(test_conversations)

    # Step 2: Load the project's Qwen model and its matching tokenizer.
    # The tokenizer converts messages into the token IDs the model processes.
    qwen, tokenizer = load_qwen_in_4bit("Qwen/Qwen3-4B-Instruct-2507")

    # Step 3: Test Qwen before training.
    # Generate from history only; use the clinician reply afterward for scoring.
    before_results = test_model(qwen, tokenizer, test_examples)

    # Step 4: Set up QLoRA fine-tuning.
    # Qwen stays frozen; the adapter learns adjustments inside Qwen.
    freeze_original_weights(qwen)
    model = attach_trainable_adapter(qwen)
    optimizer = create_optimizer(trainable_parameters(model))
    model.train()
    model.peft_config["default"].revision = "cdbee75f17c01a7cc42f958dc650907174af0554"

    # One pass through the training examples, for this simple outline.
    for conversation_so_far, clinician_reply in train_examples:
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

    # Step 5: Test again using exactly the same held-out examples and settings.
    # test_model uses evaluation mode and disables gradient calculations.
    after_results = test_model(model, tokenizer, test_examples)
    compare_results(before_results, after_results)
    # Review response quality; different wording alone is not necessarily worse.

    # Step 6: Save the adapter for use together with the same Qwen base model.
    save_adapter(model, "saved_adapter")


if __name__ == "__main__":
    fine_tuning_pseudocode()
