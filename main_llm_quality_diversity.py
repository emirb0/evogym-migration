import os
import time
import random
from collections import defaultdict

import numpy as np
import torch
from openai import OpenAI

from evogym import sample_robot, hashable
from evogym.utils import is_connected, has_actuator
from utils.algo_utils import Structure
from ppo.arguments import get_args
from LLM.run_llm import run as ppo_run


os.environ.setdefault("OPENAI_API_KEY", "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")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1/")

client = OpenAI()
_llm_quality_cache = {}


# ===========================================================
# Behavior descriptor for QD (MAP-Elites style)
# ===========================================================
def compute_behavior_descriptor(body: np.ndarray) -> np.ndarray:
    """
    body: 2D numpy array (5x5) with values in {0,1,2,3,4}
    Returns a 2D descriptor in [0,1]^2:
      d0 = fraction of actuator voxels among non-empty voxels
      d1 = normalized horizontal center-of-mass of non-empty voxels
    """
    body = np.array(body)
    mask_nonempty = body != 0
    total_voxels = np.sum(mask_nonempty)
    if total_voxels == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    # Actuators are 3 or 4
    actuator_mask = (body == 3) | (body == 4)
    num_actuators = np.sum(actuator_mask & mask_nonempty)
    frac_act = float(num_actuators) / float(total_voxels)

    coords = np.argwhere(mask_nonempty)
    if coords.size == 0:
        com_x = 0.0
    else:
        # coords[:,1] is column (x)
        com_x = float(np.mean(coords[:, 1])) / float(body.shape[1] - 1)

    return np.array([frac_act, com_x], dtype=np.float32)


def descriptor_to_cell(desc: np.ndarray, grid_shape=(10, 10)) -> tuple:
    """
    Map continuous descriptor in [0,1]^2 to integer grid cell.
    """
    desc = np.clip(desc, 0.0, 1.0)
    desc_scaled = desc * (np.array(grid_shape) - 1)
    idx = desc_scaled.astype(int)
    return int(idx[0]), int(idx[1])


# ===========================================================
# LLM-based "quality" feedback
# ===========================================================
def get_llm_quality(structure: Structure, env_name: str) -> float:
    """
    Ask the LLM to rate how promising a morphology is for the given task.
    Returns a score in [0,10]. Cached by morphology hash.
    """
    global _llm_quality_cache

    body = np.array(structure.body)
    key = hashable(body)

    if key in _llm_quality_cache:
        return _llm_quality_cache[key]

    grid_str = "\n".join(" ".join(str(int(v)) for v in row) for row in body)

    user_prompt = (
        "You are evaluating voxel-based soft robot morphologies.\n"
        f"The task is: '{env_name}'.\n\n"
        "The robot body is represented as a 5x5 grid of integers:\n"
        "0 = empty, 1 = rigid, 2 = soft, 3 = horizontal actuator, 4 = vertical actuator.\n\n"
        "Here is the morphology:\n"
        f"{grid_str}\n\n"
        "Rate how promising this morphology is for solving the task on a scale from 0 to 10.\n"
        "0 = very poor, 10 = extremely promising.\n"
        "Answer with ONLY a single number (integer or decimal)."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict morphology evaluator."},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        # Try to parse a single float
        score = float(text)
    except Exception as e:
        print(f"[WARN] LLM quality parsing failed ({e}), defaulting to 5.0")
        score = 5.0

    # Clamp to [0,10] just in case
    score = max(0.0, min(10.0, score))
    _llm_quality_cache[key] = score
    return score


# ===========================================================
# QD Archive (MAP-Elites-like)
# ===========================================================
class LLMQDArchive:
    def __init__(self, grid_shape=(10, 10), alpha=0.7, beta=0.3):
        """
        grid_shape: shape of MAP-Elites grid
        alpha: weight for environment fitness
        beta: weight for LLM quality (scaled to [0,1])
        """
        self.grid_shape = grid_shape
        self.alpha = alpha
        self.beta = beta
        # cell -> entry dict
        self.cells = {}  # (i,j) -> {"structure", "desc", "env_fitness", "llm_quality", "combined_quality"}

    def add(self, structure: Structure, env_name: str):
        desc = compute_behavior_descriptor(structure.body)
        cell = descriptor_to_cell(desc, self.grid_shape)

        env_fitness = float(structure.fitness)
        llm_quality = get_llm_quality(structure, env_name)
        combined = self.alpha * env_fitness + self.beta * (llm_quality / 10.0)

        existing = self.cells.get(cell)
        if existing is None or combined > existing["combined_quality"]:
            self.cells[cell] = {
                "structure": structure,
                "desc": desc,
                "env_fitness": env_fitness,
                "llm_quality": llm_quality,
                "combined_quality": combined,
            }

    def sample_parents(self, k: int):
        """
        Sample k parents uniformly from occupied cells.
        """
        if not self.cells:
            return []
        entries = list(self.cells.values())
        return [random.choice(entries)["structure"] for _ in range(k)]

    def best_by_env_fitness(self):
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda e: e["env_fitness"])

    def best_by_combined(self):
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda e: e["combined_quality"])


# ===========================================================
# Simple morphology mutation
# ===========================================================
def mutate_body(parent_body: np.ndarray, mutation_rate: float = 0.1, max_retries: int = 20) -> np.ndarray:
    """
    Mutate the 5x5 voxel grid by random voxel-type changes.
    Ensures connectivity and at least one actuator, up to max_retries.
    """
    parent_body = np.array(parent_body)
    for attempt in range(max_retries):
        body = parent_body.copy()
        for i in range(body.shape[0]):
            for j in range(body.shape[1]):
                if random.random() < mutation_rate:
                    body[i, j] = random.randint(0, 4)

        if is_connected(body) and has_actuator(body):
            return body

    # If all retries fail, fall back to parent body
    return parent_body.copy()


# ===========================================================
# Evaluate population via PPO pipeline
# ===========================================================
def evaluate_population(bodies, env_name: str, args, experiment_prefix: str, num_cores: int = 4):
    """
    bodies: list of 5x5 numpy arrays
    Returns: list of Structure-like objects as produced by ppo_run,
             each having .body and .fitness (and .reward).
    """
    if len(bodies) == 0:
        return []

    pop = ppo_run(
        pop_size=len(bodies),
        structure_shape=(5, 5),
        experiment_name=f"{experiment_prefix}-{int(time.time())}",
        max_evaluations=50,
        train_iters=100,
        num_cores=num_cores,
        env_name=env_name,
        args=args,
        population=bodies,
        num_generations=1,
    )
    return pop


# ===========================================================
# Main QD + LLM loop
# ===========================================================
def main():
    torch.multiprocessing.set_start_method("spawn", force=True)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = get_args()
    args.num_processes = 4
    args.cuda = True
    args.no_cuda = False
    args.eval_interval = 50

    env_name = "Carrier-v0"  # or "Walker-v0", "Pusher-v0", etc.
    max_generations = 4
    init_pop_size = 30
    num_iterations = max_generations
    offspring_per_iter = 30

    grid_shape = (10, 10)
    alpha = 0.7  # weight of env fitness
    beta = 0.3   # weight of LLM quality

    archive = LLMQDArchive(grid_shape=grid_shape, alpha=alpha, beta=beta)

    # -------------------------------------------------------
    # 1) Initial random population
    # -------------------------------------------------------
    print("[QD] Sampling initial random population...")
    population_structure_hashes = {}
    init_bodies = []

    while len(init_bodies) < init_pop_size:
        robot, _ = sample_robot((5, 5))
        if (
            is_connected(robot)
            and has_actuator(robot)
            and hashable(robot) not in population_structure_hashes
        ):
            init_bodies.append(robot)
            population_structure_hashes[hashable(robot)] = True

    init_structures = evaluate_population(
        init_bodies, env_name, args, experiment_prefix="QD-Init"
    )

    for s in init_structures:
        archive.add(s, env_name)

    print(f"[QD] Initial archive size: {len(archive.cells)}")

    # -------------------------------------------------------
    # 2) Iterative QD loop
    # -------------------------------------------------------
    for it in range(num_iterations):
        print(f"\n[QD] Iteration {it}")

        parents = archive.sample_parents(offspring_per_iter)
        if not parents:
            # If archive is empty for some reason, reuse initial population
            print("[QD] Archive empty, reusing initial population as parents.")
            parents = init_structures

        # Generate offspring by mutating parents' bodies
        new_bodies = [
            mutate_body(p.body, mutation_rate=0.1) for p in parents
        ]

        # Evaluate offspring
        offspring_structures = evaluate_population(
            new_bodies, env_name, args, experiment_prefix=f"QD-Iter{it}"
        )

        # Add offspring to archive
        for s in offspring_structures:
            archive.add(s, env_name)

        # Logging
        best_env = archive.best_by_env_fitness()
        best_comb = archive.best_by_combined()

        if best_env is not None:
            print(
                f"[QD] Archive size={len(archive.cells)}, "
                f"best_env_fitness={best_env['env_fitness']:.3f}, "
                f"best_llm_quality={best_env['llm_quality']:.3f}, "
                f"best_combined={best_env['combined_quality']:.3f}"
            )
        else:
            print("[QD] Archive is empty after iteration (unexpected).")

    print("\n[QD] Finished QD+LLM run.")
    # Here you could save archive contents, best individuals, etc.


if __name__ == "__main__":
    main()
