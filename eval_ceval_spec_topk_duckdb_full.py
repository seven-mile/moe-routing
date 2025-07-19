import os
import argparse
import math
import json
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
import duckdb


def calc_perplexity(logits, token_ids):
    assert logits.shape[:-1] == token_ids.shape, \
        f"Logits shape {logits.shape} does not match token_ids shape {token_ids.shape}"
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        token_ids.view(-1),
        reduction='none'
    )
    return torch.exp(loss).view(token_ids.shape)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate causal LM on CEval with speculative sampling and record to DuckDB"
    )
    parser.add_argument("--dataset_config", required=True,
                        help="CEval dataset config (e.g., computer_architecture)")
    parser.add_argument("--experiment_name", default=None,
                        help="Logical name for this experiment group")
    parser.add_argument("--model_name", default="Qwen/Qwen3-30B-A3B",
                        help="Pretrained model identifier")
    parser.add_argument("--split", default="val",
                        help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (None for all)")
    parser.add_argument("--max_think_tokens", type=int, default=4096,
                        help="Max tokens for think phase")
    parser.add_argument("--step_tokens", type=int, default=4,
                        help="Tokens generated per speculative step")
    parser.add_argument("--duckdb_path", default="data/ceval.duckdb",
                        help="Path to output DuckDB file")
    parser.add_argument("--cuda_visible_devices", default="0",
                        help="CUDA_VISIBLE_DEVICES setting")
    return parser.parse_args()


def init_duckdb(db_path):
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_experiment_id START WITH 1;
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id INTEGER PRIMARY KEY DEFAULT nextval('seq_experiment_id'),
            name TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT now()
        )""")
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_run_id START WITH 1;
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY DEFAULT nextval('seq_run_id'),
            experiment_id INTEGER REFERENCES experiments(experiment_id),
            started_at TIMESTAMP DEFAULT now(),
            finished_at TIMESTAMP,
            status TEXT,
            notes TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval (
            sample_idx INTEGER,
            run_id INTEGER REFERENCES runs(run_id),
            question TEXT,
            think_text TEXT,
            predicted_answer TEXT,
            ground_truth TEXT,
            spec_token_ppls TEXT,
            spec_accept_counts TEXT,
            token_topks TEXT,
            timestamp DOUBLE
        )""")
    return conn


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # 如果 experiment_name 未指定，则使用默认格式
    if args.experiment_name is None:
        args.experiment_name = f"ours_ceval_{args.dataset_config}_{args.split}"

    # 初始化数据库与记录实验/运行信息
    conn = init_duckdb(args.duckdb_path)
    # 获取或插入 experiment
    exp = conn.execute(
        "SELECT experiment_id FROM experiments WHERE name = ?", [args.experiment_name]
    ).fetchone()
    if exp:
        experiment_id = exp[0]
    else:
        experiment_id = conn.execute(
            "INSERT INTO experiments(name) VALUES(?) RETURNING experiment_id", [args.experiment_name]
        ).fetchone()[0]
    # 插入新的 run
    run_id = conn.execute(
        "INSERT INTO runs(experiment_id, status) VALUES(?, 'RUNNING') RETURNING run_id", [experiment_id]
    ).fetchone()[0]

    print(f"Experiment {args.experiment_name} (ID={experiment_id}), run ID={run_id}")

    print(f"Loading model {args.model_name} on GPU(s) {args.cuda_visible_devices}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    device = next(model.parameters()).device
    print("Model loaded.\n")

    print(f"Loading dataset ceval/ceval-exam ({args.dataset_config}) split={args.split}...")
    ds = load_dataset("ceval/ceval-exam", args.dataset_config, split=args.split)
    if args.num_samples:
        ds = ds.select(range(args.num_samples))
    print(f"Loaded {len(ds)} samples.\n")

    think_stop = "</think>"
    options = ["A","B","C","D"]
    correct = 0

    for idx, doc in enumerate(tqdm(ds, desc="Evaluating")):
        prompt_start = (
            f"{doc['question'].strip()}\n"
            f"A. {doc['A']}\n"
            f"B. {doc['B']}\n"
            f"C. {doc['C']}\n"
            f"D. {doc['D']}\n"
            f"<think>"
        )
        input_ids = tokenizer(prompt_start, return_tensors="pt").input_ids.to(device)

        # init cache
        with torch.no_grad():
            out = model(input_ids[:, :-1], use_cache=True)
            past = out.past_key_values

        gen_ids = []
        spec_ppls = []      # store only accepted-token PPLs
        spec_accept = []
        total_topks = []

        cur_ids = input_ids
        base_k = model.config.num_experts_per_tok

        avg_accepts = []
        avg_topks = []

        # speculative loop
        with tqdm(total=args.max_think_tokens, desc="Thinking", leave=False) as pbar:
            while len(gen_ids) < args.max_think_tokens:
                cur_len = cur_ids.size(-1)
                with torch.no_grad():
                    gen_out = model.generate(
                        cur_ids,
                        max_new_tokens=args.step_tokens,
                        output_logits=True,
                        return_dict_in_generate=True,
                        past_key_values=past,
                        use_cache=True
                    )
                new_ids = gen_out.sequences[0, cur_ids.size(-1):]
                logits = torch.cat(gen_out.logits, dim=0)
                ppls = calc_perplexity(logits, new_ids)

                # determine allowed top_k per token
                topks = torch.full_like(new_ids, base_k, dtype=torch.int32)
                # topks = torch.where(ppls > 2.0, base_k, base_k - 1)
                topks = torch.where(ppls < 2.0, base_k - 1, topks)
                topks = torch.where(ppls < 1.02, base_k - 2, topks)
                topks = torch.where(ppls < 1.004, base_k - 3, topks)

                # verify
                past.crop(cur_len-1)
                verify_ids = torch.cat([cur_ids[:,-1:], new_ids[None, :-1]], dim=1)
                with torch.no_grad():
                    verify_out = model(verify_ids, use_cache=True, past_key_values=past, token_top_ks=topks)

                # count accepted tokens and their PPLs
                accepted_ids = []
                step_accepted_ppls = []
                for i, (nid, logit) in enumerate(zip(new_ids, verify_out.logits[0])):
                    pick = int(logit.argmax(-1))
                    accepted_ids.append(pick)
                    step_accepted_ppls.append(ppls[i].item())
                    if pick != nid:
                        # truncate kvcache to the last accepted token
                        past.crop(cur_len + i)
                        break
                accepted = len(step_accepted_ppls)
                spec_accept.append(accepted)

                # record only PPLs of accepted tokens
                spec_ppls.extend(step_accepted_ppls)
                # record token top_k
                total_topks.extend(topks[:accepted].tolist())

                # advance
                gen_ids.extend(accepted_ids)
                cur_ids = torch.cat([cur_ids, torch.tensor([accepted_ids], device=device)], dim=1)
                pbar.update(len(accepted_ids))

                text = tokenizer.decode(gen_ids)
                if think_stop in text:
                    gen_ids = tokenizer.encode(text.split(think_stop)[0]+think_stop, add_special_tokens=False)
                    break

        think_text = tokenizer.decode(gen_ids)

        # final prompt
        full_text = prompt_start + think_text + "\n答案："
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            out = model(full_ids)
            last_logits = out.logits[0,-1]
            log_probs = F.log_softmax(last_logits, dim=-1)
        llhs = {opt: log_probs[tokenizer.encode(opt)].item() for opt in options}
        pred = max(llhs, key=llhs.get)
        truth = doc['answer']
        if pred == truth:
            correct += 1

        # 插入 eval 记录时带上 run_id
        conn.execute(
            "INSERT INTO eval VALUES (?,?,?,?,?,?,?,?,?,?)",
            [idx, run_id,
             prompt_start, think_text, pred, truth,
             json.dumps(spec_ppls), json.dumps(spec_accept),
             json.dumps(total_topks), time.time()]
        )
        conn.commit()

        avg_accept = sum(spec_accept)/len(spec_accept) if spec_accept else 0
        avg_topk = sum(total_topks)/base_k/len(total_topks)*100 if total_topks else 0
        print(f"#{idx} pred={pred} true={truth} accept_avg={avg_accept:.2f} topk_avg%={avg_topk:.2f}%")
        avg_accepts.extend(spec_accept)
        avg_topks.extend(total_topks)

    print(f"Done. acc={correct}/{len(ds)} total_accept_avg={sum(avg_accepts)/len(avg_accepts):.2f} total_topk_avg%={sum(avg_topks)/len(avg_topks):.2f}%")

    # 更新 run 状态
    conn.execute(
        "UPDATE runs SET finished_at = now(), status = 'COMPLETED' WHERE run_id = ?", [run_id]
    )
    conn.commit()

if __name__ == "__main__":
    main()
