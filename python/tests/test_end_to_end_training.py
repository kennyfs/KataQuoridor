import os
import sys
import shutil
import tempfile
import subprocess
import numpy as np
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_root, "python"))

from katago.train import modelconfigs
from katago.train.model_pytorch import Model
from katago.train.metrics_pytorch import Metrics
from katago.train.data_processing_pytorch import read_npz_training_data


def test_end_to_end_training():
    katago_bin = os.path.join(repo_root, "build", "katago")
    assert os.path.exists(katago_bin), f"KataGo binary not found at {katago_bin}. Please run make -C build first."

    shuffle_script = os.path.join(repo_root, "python", "shuffle.py")
    assert os.path.exists(shuffle_script), f"shuffle.py not found at {shuffle_script}"

    with tempfile.TemporaryDirectory() as base_temp_dir:
        raw_dir = os.path.join(base_temp_dir, "raw")
        shuffled_dir = os.path.join(base_temp_dir, "shuffled")
        tmp_dir = os.path.join(base_temp_dir, "tmp")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(shuffled_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)

        # 1. Generate sample Quoridor selfplay training data via C++ subcommand
        num_files = 4
        rows_per_file = 32
        print(f"[E2E] Generating {num_files} raw files with {rows_per_file} rows each...")
        cmd_gen = [katago_bin, "writesampletrainquoridor", raw_dir, str(num_files), str(rows_per_file)]
        res_gen = subprocess.run(cmd_gen, cwd=repo_root, capture_output=True, text=True)
        assert res_gen.returncode == 0, f"writesampletrainquoridor failed:\nSTDOUT:\n{res_gen.stdout}\nSTDERR:\n{res_gen.stderr}"

        raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".npz")]
        assert len(raw_files) == num_files, f"Expected {num_files} raw files, found {len(raw_files)}"

        # Verify raw npz shapes and contents
        sample_npz = np.load(raw_files[0])
        print(f"[E2E] Verifying raw npz shapes: {list(sample_npz.keys())}")
        assert sample_npz["binaryInputNCHWPacked"].shape == (rows_per_file, 17, 11)
        assert sample_npz["globalInputNC"].shape == (rows_per_file, 15)
        assert sample_npz["policyTargetsNCMove"].shape == (rows_per_file, 2, 243)
        assert sample_npz["globalTargetsNC"].shape == (rows_per_file, 80)
        assert sample_npz["scoreDistrN"].shape == (rows_per_file, 282)
        assert sample_npz["valueTargetsNCHW"].shape == (rows_per_file, 4, 9, 9)
        assert sample_npz["qValueTargetsNCMove"].shape == (rows_per_file, 3, 243)

        # 2. Run shuffle.py on raw data
        print("[E2E] Running shuffle.py...")
        cmd_shuffle = [
            sys.executable,
            shuffle_script,
            "-min-rows", "1",
            "-keep-target-rows", "all",
            "-num-processes", "1",
            "-approx-rows-per-out-file", "32",
            "-out-dir", shuffled_dir,
            "-out-tmp-dir", tmp_dir,
            raw_dir,
        ]
        res_shuffle = subprocess.run(cmd_shuffle, cwd=repo_root, capture_output=True, text=True)
        assert res_shuffle.returncode == 0, f"shuffle.py failed:\nSTDOUT:\n{res_shuffle.stdout}\nSTDERR:\n{res_shuffle.stderr}"

        shuffled_files = [os.path.join(shuffled_dir, f) for f in os.listdir(shuffled_dir) if f.endswith(".npz")]
        assert len(shuffled_files) > 0, "No shuffled .npz files found!"
        print(f"[E2E] Successfully generated {len(shuffled_files)} shuffled files.")

        # 3. Setup Model, Optimizer, and Metrics
        cfg = modelconfigs.base_config_of_name["b2c64_quoridor"]
        model = Model(cfg, pos_len=9)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = Metrics(world_size=1, raw_model=model)

        # 4. Load shuffled batches via read_npz_training_data
        batch_size = 16
        data_gen = read_npz_training_data(
            npz_files=shuffled_files,
            batch_size=batch_size,
            world_size=1,
            rank=0,
            pos_len=9,
            device="cpu",
            randomize_symmetries=True,
            include_meta=False,
            model_config=cfg,
            prefetch_depth=1,
        )

        batches = list(data_gen)
        assert len(batches) > 0, "Failed to load any batches from shuffled files!"
        print(f"[E2E] Loaded {len(batches)} batches of size {batch_size} from shuffled data.")

        # 5. Overfitting test on batch 0 over 5 gradient descent steps
        print("[E2E] Running 5-step optimization check on batch 0...")
        batch0 = batches[0]
        losses = []

        for step in range(5):
            optimizer.zero_grad()
            out_byheads = model(batch0["binaryInputNCHW"], batch0["globalInputNC"])
            post = model.postprocess_output(out_byheads)

            results = metrics.metrics_dict_batchwise(
                raw_model=model,
                model_output_postprocessed_byheads=post,
                extra_outputs=None,
                batch=batch0,
                is_training=True,
                soft_policy_weight_scale=1.0,
                disable_optimistic_policy=False,
                meta_kata_only_soft_policy=False,
                value_loss_scale=1.5,
                td_value_loss_scales=[0.2, 0.2, 0.2, 0.2],
                seki_loss_scale=1.0,
                variance_time_loss_scale=1.0,
                main_loss_scale=1.0,
                intermediate_loss_scale=0.25,
                include_model_norms=True,
            )

            loss = results["loss_sum"]
            assert torch.isfinite(loss).item(), f"Loss is not finite at step {step}: {loss}"
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
            print(f"  Step {step}: loss_sum = {loss_val:.4f}")

        assert losses[-1] < losses[0], f"Loss did not decrease after 5 steps! {losses}"
        print(f"[E2E] Loss strictly decreased from {losses[0]:.4f} to {losses[-1]:.4f}.")

        # 6. Run remaining batches to ensure robust forward/backward across all shuffled samples
        print(f"[E2E] Running 1 step forward/backward on remaining {len(batches) - 1} batches...")
        for i, batch in enumerate(batches[1:], start=1):
            optimizer.zero_grad()
            out_byheads = model(batch["binaryInputNCHW"], batch["globalInputNC"])
            post = model.postprocess_output(out_byheads)

            results = metrics.metrics_dict_batchwise(
                raw_model=model,
                model_output_postprocessed_byheads=post,
                extra_outputs=None,
                batch=batch,
                is_training=True,
                soft_policy_weight_scale=1.0,
                disable_optimistic_policy=False,
                meta_kata_only_soft_policy=False,
                value_loss_scale=1.5,
                td_value_loss_scales=[0.2, 0.2, 0.2, 0.2],
                seki_loss_scale=1.0,
                variance_time_loss_scale=1.0,
                main_loss_scale=1.0,
                intermediate_loss_scale=0.25,
                include_model_norms=True,
            )
            loss = results["loss_sum"]
            assert torch.isfinite(loss).item(), f"Batch {i} loss not finite: {loss}"
            loss.backward()
            optimizer.step()

        print("[E2E] All batches trained successfully without NaN or gradient issues.")


if __name__ == "__main__":
    test_end_to_end_training()
    print("\n>>> End-to-End Training Pipeline Test PASSED! <<<")
