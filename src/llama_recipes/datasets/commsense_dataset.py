import copy
import datasets


def get_preprocessed_commsense(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("json", data_files='src/llama_recipes/datasets/CommSense_CoT_2000.json', field=split, split='train')
    def apply_prompt_template(sample):
        return {
            "question": sample["question"],
            "answer": sample["answer"],
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        question = tokenizer.encode(tokenizer.bos_token + sample["question"], add_special_tokens=False)
        answer = tokenizer.encode(sample["answer"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": question + answer,
            "attention_mask" : [1] * (len(question) + len(answer)),
            "labels": [-100] * len(question) + answer,
        }
        return sample
    
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
