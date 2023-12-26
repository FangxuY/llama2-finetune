import copy
import datasets


def get_commsense_dataset(dataset_config, tokenizer, split):
    if split == 'train':
        dataset = datasets.load_from_disk("json", data_files='./CommSense.jsonl')
    else:
        dataset = datasets.load_from_disk("json", data_files='./CommSense_eval.jsonl')
    def apply_prompt_template(sample):
        return {
            "question": sample["question"],
            "answer": sample["answer"],
        }

    dataset = dataset.map(apply_prompt_template)

    def tokenize_add_label(sample):
        question = tokenizer.encode(tokenizer.bos_token + sample["question"], add_special_tokens=False)
        answer = tokenizer.encode(sample["answer"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": question + answer,
            "attention_mask" : [1] * (len(question) + len(answer)),
            "labels": [-100] * len(answer) + question,
            }

        return sample
    
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset