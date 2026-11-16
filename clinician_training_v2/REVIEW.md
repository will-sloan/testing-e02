# Review of the original training implementation

The original implementation is a useful small demonstration of QLoRA. It correctly freezes the base model, applies Qwen's chat template, masks prompt labels, retains the target end marker, and groups sessions by participant when `participant_id` is present. Those are worth retaining.

The implementation and available synthetic samples were inspected. The real conversations, trained adapter, GPU runtime, and failing generation logs were not available. The findings below identify concrete weaknesses; they do not establish a single proven cause of the reported degeneration.

## What can explain poor before and after responses

1. **The baseline uses the same transcript-derived inputs.** A poor before-training response cannot be attributed to the newly trained adapter. Speaker mapping, partial turns, ambiguous history, the system prompt, quantized runtime, and generation must be evaluated independently of optimizer changes.
2. **Each clinician message is assumed to be a suitable target.** `create_examples.py` neither joins adjacent speaker fragments nor requires the preceding message to be from the patient. If transcription chunks divide one clinician turn, the target can be an arbitrary continuation. The desired output is spoken clinician-style language, so natural fillers and brief acknowledgments are appropriate; transcription errors and pathological repetition are not.
3. **Selection favors the first eligible replies.** `MAX_SAMPLES` limits both loaded conversations and generated examples. With the checked-in value of eight, the code permits six training targets and two test targets, potentially concentrated in one conversation each. Larger values still retain the first eligible examples rather than sampling them across the whole corpus.
4. **Context loss can remove evidence needed for the answer.** The recent-window fitter retains the latest turns but can remove earlier goals or clinical details. The revised fitter avoids cutting targets, but it cannot restore omitted context. A long advertised model context window does not establish that a particular GPU can fine-tune at that length.
5. **The comparison is weak.** The original test routine samples once without resetting paired evaluation seeds, reports only decoded text and one end-token check, and has no separate validation set. The combined warning does not identify which output reached the ceiling. The generation configuration may recognize more EOS IDs than the tokenizer-only check.
6. **Training has few controls for deterioration.** The loop uses one example per optimizer update, one pass, no shuffled trainer sampler, no accumulation, no schedule, no clipping, no intermediate validation, and no best-checkpoint selection. Its loss measures prediction of supplied target tokens; it is not a free-generation quality score.
7. **Experiment reproduction is incomplete.** The adapter save omits a complete experiment manifest linking settings, data, dependencies, and evaluation. Repeated runs can overwrite the same adapter folder.

## Replacement design

Version 2 uses transcript normalization, patient → clinician turn boundaries, conservative quality checks, optional explicit target review, participant-aware splits, and duplicate checks before training. It keeps the current non-thinking Qwen checkpoint and targets spoken clinician-style replies. The user confirmed that each file represents a separate patient, so missing IDs fall back to source-file grouping, not one group per JSONL record. Stable participant IDs become necessary when visits for one patient span files.

The training method is completion-only supervised fine-tuning using the current Transformers Trainer and PEFT implementations. Explicit labels provide an inspectable contract: history and padding have label `-100`; the final target response and its end-of-message marker remain supervised. This is the same completion-only objective supported by TRL, without relying on automatic template rewriting. TRL's assistant-only masking requires generation tags in the template, while prompt/completion training supports completion-only loss directly. [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer)

The memory strategy is NF4 4-bit quantization with double quantization, gradient checkpointing, and small microbatches. Current bitsandbytes supports Windows NVIDIA hardware, subject to its software and device requirements. [Transformers quantization guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes), [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/installation)

Rank-stabilized LoRA is a supported modern adapter variant. Its scale is `alpha / sqrt(rank)`, rather than conventional LoRA's `alpha / rank`; the default rank 16 and alpha 8 give scale 2. This avoids accidentally quadrupling the scale by carrying the old rank-16/alpha-32 combination into rsLoRA. All-linear targeting adapts attention and feed-forward projections while excluding the language-model output layer. Zero-initialized LoRA B preserves the base function before training. [PEFT LoRA reference](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)

Gradient accumulation increases the effective batch without requiring all examples in memory at once. Current Trainer and causal-LM loss support normalization over supervised prediction tokens across accumulated microbatches. The implementation should retain that path rather than averaging already-averaged variable-length target losses by hand. Validation loss is useful for checkpoint selection, but generation and human review remain necessary. [Transformers gradient accumulation](https://github.com/huggingface/transformers/blob/main/docs/source/en/grad_accumulation.md), [Trainer reference](https://huggingface.co/docs/transformers/main_classes/trainer)

## Choices that require evidence before adoption

- **DPO or reinforcement learning:** Ordinary transcripts provide neither chosen/rejected response pairs nor a trustworthy reward. Build clinician-reviewed preferences first. A superficially shorter answer is not a sufficient reward.
- **Automatic transcript rewriting:** A separate language model can introduce facts, erase uncertainty, or change advice. This workflow makes target edits explicit and reviewable instead of treating generated rewrites as ground truth.
- **Synthetic historical summaries:** Summaries can help with long context, but must be made only from information available before the target turn. Full-session summaries leak the future. Version 2 does not generate summaries automatically.
- **Sequence packing:** It can improve throughput, but boundary isolation and label masks need verification. Current TRL best-fit packing enables padding-free execution, which requires FlashAttention 2 or 3. The Windows-first default uses separate examples instead. [TRL packing documentation](https://huggingface.co/docs/trl/sft_trainer#packing)
- **DoRA, EVA, PiSSA, and additional adapter variants:** These are useful research options, not substitutes for a correct dataset and baseline. They add initialization or runtime tradeoffs. rsLoRA is the starting controlled experiment, not a claim that it is universally best.
- **Unsloth:** It offers supported Windows installation and efficiency improvements. It is a reasonable later backend experiment after the reference pipeline is validated. Adding its patching and dependencies to the first debugging baseline would make attribution harder. [Unsloth Windows installation](https://unsloth.ai/docs/get-started/install/windows-installation)

## How to decide whether it worked

Start with an untrained baseline on simple sanity prompts and validation histories. Resolve obvious repetition or formatting failures before relying on training. Then compare the best validation checkpoint with the same base checkpoint using identical prompts, generation settings, and per-example seeds. Record output length, termination reason, repetition indicators, and reference loss, while retaining generated text for human review.

Human review should ask: Did the response address the patient's latest point? Did it accurately use earlier context? Did it preserve uncertainty and avoid invented facts? Is it a complete, useful next turn? Is any advice clinically appropriate? Does it finish naturally without filler or loops? A short, naturally terminated answer can still fail every substantive criterion.

Select settings using validation. Use the held-out patient test set only after selecting the recipe. Report the number of patients and examples, excluded long-context examples, and rejected targets alongside quality results so an apparent gain cannot hide reduced dataset coverage.

## Runtime versions

The researched releases available on 2026-11-16 were Transformers 5.23.0, PEFT 0.21.3, Accelerate 1.15.0, and bitsandbytes 0.50.2. They are available releases, not a claim that this exact Windows/CUDA combination was GPU-tested during development. Python 3.11 or newer is required by this Transformers release. Install the appropriate CUDA PyTorch build separately and save the environment details with each run. [Transformers release](https://pypi.org/project/transformers/5.23.0/), [PEFT release](https://pypi.org/project/peft/0.21.3/), [Accelerate release](https://pypi.org/project/accelerate/1.15.0/), [bitsandbytes release](https://pypi.org/project/bitsandbytes/0.50.2/)
